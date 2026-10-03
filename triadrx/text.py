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
