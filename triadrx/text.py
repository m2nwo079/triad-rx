"""Text normalization and vocabulary matching shared by vocabulary building and abstract matching.

Both sides must use these functions so that a vocabulary surface and the text it should
match are transformed in exactly the same way.
"""
import re
import unicodedata
from collections import defaultdict

# Tokens are runs of letters and digits; every other character separates tokens
TOKEN = re.compile(r"[A-Za-z0-9]+")


def norm(text: str) -> str:
    """Rule R2: NFKC, lowercase, unify hyphens, underscores, slashes and whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\-_/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,;:")


def ci_tokens(text: str) -> tuple[str, ...]:
    """Tokens for case-insensitive matching, taken from the normalized text."""
    return tuple(TOKEN.findall(norm(text)))


def cs_tokens(text: str) -> tuple[str, ...]:
    """Tokens for case-sensitive acronym matching, taken from the NFKC text without lowercasing."""
    return tuple(TOKEN.findall(unicodedata.normalize("NFKC", text)))


class Matcher:
    """Longest-match, non-overlapping n-gram matcher over vocabulary surfaces.

    Case-insensitive ("ci") and case-sensitive ("cs") surfaces are matched in separate
    passes. A token sequence that maps to more than one term is ambiguous and is skipped;
    such sequences are listed in ``ambiguous`` for reporting.
    """

    def __init__(self, terms: list[dict], statuses: tuple[str, ...] = ("active",)):
        owners = {"ci": defaultdict(set), "cs": defaultdict(set)}
        for term in terms:
            if term["status"] not in statuses:
                continue
            for surface in term["surfaces"]:
                mode = surface["match"]
                tokens = ci_tokens(surface["text"]) if mode == "ci" else cs_tokens(surface["text"])
                if tokens:
                    owners[mode][tokens].add(term["id"])
        self.index = {}
        self.ambiguous = {}
        # Candidate surface lengths keyed by first token, longest first, to keep scanning fast
        self.lengths = {}
        for mode, table in owners.items():
            self.index[mode] = {k: next(iter(v)) for k, v in table.items() if len(v) == 1}
            self.ambiguous[mode] = {k: sorted(v) for k, v in table.items() if len(v) > 1}
            by_first = defaultdict(set)
            for key in self.index[mode]:
                by_first[key[0]].add(len(key))
            self.lengths[mode] = {k: sorted(v, reverse=True) for k, v in by_first.items()}

    def _scan(self, tokens: tuple[str, ...], mode: str) -> list[tuple[int, int, str]]:
        index, lengths = self.index[mode], self.lengths[mode]
        found, i = [], 0
        while i < len(tokens):
            for n in lengths.get(tokens[i], ()):
                if i + n <= len(tokens):
                    tid = index.get(tokens[i : i + n])
                    if tid is not None:
                        found.append((i, i + n, tid))
                        i += n
                        break
            else:
                i += 1
        return found

    def spans(self, text: str) -> list[dict]:
        """All matches with mode, token span and term id."""
        out = []
        for mode, tokens in (("ci", ci_tokens(text)), ("cs", cs_tokens(text))):
            for start, end, tid in self._scan(tokens, mode):
                out.append({"mode": mode, "start": start, "end": end, "id": tid, "tokens": " ".join(tokens[start:end])})
        return out

    def terms(self, text: str) -> set[str]:
        """Set of matched term ids."""
        return {m["id"] for m in self.spans(text)}


def refine_terms(terms: list[dict], skip_chars: list[str], excluded: set[tuple[str, str]] = frozenset()) -> tuple[list[dict], set[str]]:
    """Terms without surfaces that hold a character the tokenizer drops or that were excluded by
    review ((term id, surface) pairs), and the ids of terms left with only case-sensitive surfaces."""
    out = []
    for term in terms:
        surfaces = [s for s in term["surfaces"]
                    if not any(c in s["text"] for c in skip_chars) and (term["id"], s["text"]) not in excluded]
        if surfaces:
            out.append({**term, "surfaces": surfaces})
    return out, {t["id"] for t in out if all(s["match"] == "cs" for s in t["surfaces"])}


def refined_spans(matcher: Matcher, texts: list[str], cs_only: set[str], rules: dict,
                  exclusions: dict[str, list[tuple[str, ...]]] | None = None) -> list[list[dict]]:
    """Matches of each text of one paper after the refined matching rules.

    - drop_nested: a match whose tokens lie inside a longer match of the other pass is dropped
    - exclusions: a match of a term is dropped when one of the term's excluded phrases (token
      tuples) covers it
    - acronym_needs_full_name: a case-sensitive match counts only when the same term also has a
      case-insensitive match somewhere in the paper's texts, unless the term has no
      case-insensitive surface
    """
    exclusions = exclusions or {}
    kept = []
    for text in texts:
        spans = matcher.spans(text)
        if rules["drop_nested"]:
            spans = [m for m in spans if not any(o["start"] <= m["start"] and m["end"] <= o["end"]
                                                 and o["end"] - o["start"] > m["end"] - m["start"] for o in spans)]
        if exclusions:
            tokens = ci_tokens(text)
            spans = [m for m in spans if not any(_covers(tokens, phrase, m["start"], m["end"]) for phrase in exclusions.get(m["id"], ()))]
        kept.append(spans)
    full = {m["id"] for spans in kept for m in spans if m["mode"] == "ci"}
    return [[m for m in spans if m["mode"] == "ci" or not rules["acronym_needs_full_name"]
             or m["id"] in cs_only or m["id"] in full] for spans in kept]


def _covers(tokens: tuple[str, ...], phrase: tuple[str, ...], start: int, end: int) -> bool:
    """True when `phrase` occurs in `tokens` at a position whose span contains [start, end)."""
    n = len(phrase)
    for p in range(max(0, end - n), min(start, len(tokens) - n) + 1):
        if tokens[p:p + n] == phrase:
            return True
    return False
