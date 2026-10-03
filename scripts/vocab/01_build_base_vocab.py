"""Build the time-independent base vocabulary (rules R1-R3, R4 candidates, restoration) from the PWC archive."""
import collections
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.text import norm

DATA = Path("data/scope")
STUDY = Path("config/study.json")
OUT_VOCAB = Path("vocab/vocab_base.json")
OUT_RESULT = Path("results/vocab_base.json")
PAPER_FILES = [DATA / f"pwa_{i}.parquet" for i in range(4)]

# R1: spam signals checked on each surface form separately, because spam
# sometimes overwrote only the full_name of a legitimate method
SPAM_KEYWORDS = (
    r"contact|customer (service|support|care)|refunds?|refundable|cancell?ations?|"
    r"reservations?|booking|flights?|airlines?|cruises?|phone|whatsapp|wallet|promo|"
    r"cancel|subscription|antivirus|helpline|helpdesk|ways to call|24 7|quickbooks|faqs?|complain|dispute|step by step|speak to|talk to|live agent|help and support|"
    r"travel|expedia|robinhood|coinspot|kraken|caribbean|lufthansa|klm|emirates|"
    r"iberia|qatar|southwest|fidelity|vanguard|blackrock|schwab|comment|c[oó]mo"
)
# Structural spam signals apply to every name
SPAM_STRUCTURE = re.compile(
    r"[?¿{}【】〗™®☎]|\[\[|\d{6,}|\d{3}\D{1,3}\d{3}\D{1,3}\d{4}"
    r"|^[^A-Za-z]*(how|what|who|does|do|can|is|are|why|where|when)\b"
    r"|[\U00010000-\U0010FFFF]",
    re.I,
)
# Keyword signals apply to method names only, because real task names contain
# words such as comment, contact or travel (for example "Toxic Comment Classification")
SPAM_KEYWORD = re.compile(r"\b(" + SPAM_KEYWORDS + r")\b", re.I)


def has_symbol(text: str) -> bool:
    # Emoji and decorative symbols (Unicode category So) do not appear in method or task names
    return any(unicodedata.category(ch) == "So" for ch in text)


def is_spam(text: str, use_keywords: bool) -> bool:
    if SPAM_STRUCTURE.search(text) or has_symbol(text):
        return True
    return use_keywords and bool(SPAM_KEYWORD.search(text))


ACRONYM = re.compile(r"^[A-Z][A-Z0-9\-\.]*$")
PAREN = re.compile(r"^(?P<base>.*?)\s*\((?P<inner>[^()]*)\)\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def split_paren(text: str) -> tuple[str, str | None]:
    # R2: "Reinforcement Learning (RL)" -> ("Reinforcement Learning", "RL")
    match = PAREN.match(text.strip())
    if match and match.group("base"):
        return match.group("base"), match.group("inner").strip()
    return text, None


STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with", "via", "by"}


def initial_variants(phrase: str) -> set[str]:
    # Candidate acronyms from an expansion: each space-separated word gives its first
    # letter, or the first letters of its hyphen parts; stopwords may be skipped
    variants = {""}
    for word in re.split(r"\s+", phrase.strip()):
        parts = [p for p in re.split(r"[\-/]", word) if re.search(r"[A-Za-z0-9]", p)]
        if not parts:
            continue
        options = {re.sub(r"[^A-Za-z0-9]", "", parts[0])[:1].upper()}
        options.add("".join(re.sub(r"[^A-Za-z0-9]", "", q)[:1] for q in parts).upper())
        if word.lower() in STOPWORDS:
            options.add("")
        variants = {v + o for v in variants for o in options if len(v + o) <= 12}
    return variants


def letters(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


class Vocab:
    def __init__(self, wordlist: set[str], cfg: dict):
        self.wordlist = wordlist
        self.min_len = cfg["r3_min_acronym_len"]
        self.terms = {}
        self.removed = []
        self.restored = []

    def remove(self, source: str, text: str, rule: str, reason: str) -> None:
        self.removed.append({"source": source, "text": text, "rule": rule, "reason": reason})

    def term(self, ttype: str, key: str) -> dict:
        tid = f"{ttype}:{key}"
        if tid not in self.terms:
            self.terms[tid] = {"id": tid, "type": ttype, "key": key, "surfaces": {}, "sources": set()}
        return self.terms[tid]

    def add_acronym(self, term: dict, acronym: str, source: str) -> None:
        # R3: acronyms of at least min_len letters are matched case-sensitively.
        # No dictionary check: case-sensitive matching does not collide with ordinary
        # words, and a dictionary check removed key models such as BERT and RAG.
        # False positives from all-caps emphasis are measured in M1.
        if len(letters(acronym)) < self.min_len:
            self.remove(source, acronym, "R3", "acronym_too_short")
            return
        term["surfaces"][acronym] = "cs"

    def add_method(self, name: str, full_name: str | None, source: str) -> None:
        name_ok = isinstance(name, str) and name.strip() and not is_spam(name, use_keywords=True)
        full_ok = isinstance(full_name, str) and full_name.strip() and not is_spam(full_name, use_keywords=True)
        if isinstance(name, str) and not name_ok:
            self.remove(source, name, "R1", "spam_pattern")
        if isinstance(full_name, str) and full_name.strip() and not full_ok and full_name != name:
            self.remove(source, full_name, "R1", "spam_pattern")
        if not name_ok:
            return
        name = name.strip()
        base, inner = split_paren(full_name.strip()) if full_ok else (None, None)
        if ACRONYM.match(name):
            key_text = base if base and " " in base else name
            term = self.term("M", norm(key_text))
            self.add_acronym(term, name, source)
        else:
            term = self.term("M", norm(name))
            term["surfaces"][norm(name)] = "ci"
        if base and " " in base:
            term["surfaces"][norm(base)] = "ci"
        if inner and ACRONYM.match(inner):
            self.add_acronym(term, inner, source)
        term["sources"].add(source)

    def add_task(self, task: str) -> None:
        source = "pwc_paper_tasks"
        if not isinstance(task, str) or not task.strip():
            return
        if is_spam(task, use_keywords=False):
            self.remove(source, task, "R1", "spam_pattern")
            return
        base, inner = split_paren(task)
        key = norm(base)
        if not key:
            return
        term = self.term("T", key)
        term["surfaces"][key] = "ci"
        if inner and ACRONYM.match(inner):
            self.add_acronym(term, inner, source)
        term["sources"].add(source)

    def restore_from_composites(self, pairs: set[tuple[str, str]], min_support: int) -> None:
        # Restoration rule: an acronym lost to spam is restored when it is spelled out as a
        # building block of other clean method records. Example: "U-Net GAN", "MSGAN" and
        # "PresGAN" all end with GAN and their full names contain "Generative Adversarial
        # Network". Evidence counts distinct name tokens, so repeated spam names do not add up.
        # Only names are used, and existing surfaces are never overwritten.
        existing = {s for term in self.terms.values() for s in term["surfaces"]}
        evidence = collections.defaultdict(set)
        for name, full_name in pairs:
            if not (isinstance(name, str) and isinstance(full_name, str)):
                continue
            if is_spam(name, use_keywords=True) or is_spam(full_name, use_keywords=True):
                continue
            words = re.split(r"\s+", full_name.strip())
            for token in re.split(r"\s+", name.strip()):
                run = re.search(r"[A-Z]+$", token)
                if not run:
                    continue
                tail = run.group(0)
                # (acronym, end word index, span) for every suffix of the trailing capitals
                matches = []
                for start in range(len(tail) - self.min_len + 1):
                    acronym = tail[start:]
                    for i in range(len(words)):
                        for j in range(i + 1, min(len(words), i + 8) + 1):
                            span_words = words[i:j]
                            if span_words[0].lower() in STOPWORDS or span_words[-1].lower() in STOPWORDS:
                                continue
                            span = " ".join(span_words)
                            if not span.isupper() and acronym in initial_variants(span):
                                matches.append((acronym, j, span))
                # A suffix is only a fragment when a longer suffix of the same token
                # matches a span ending at the same word (LIP inside CLIP, STM inside LSTM)
                for acronym, end, span in matches:
                    if any(len(a) > len(acronym) and e == end for a, e, _ in matches):
                        continue
                    if acronym in existing:
                        continue
                    expansion = norm(span)
                    # Key plural and singular spellings together
                    if expansion.endswith("s") and len(expansion.split()[-1]) > 4:
                        expansion = expansion[:-1]
                    evidence[(acronym, expansion)].add(token)
        for (acronym, expansion), tokens in sorted(evidence.items()):
            if len(tokens) < min_support or expansion in existing:
                continue
            term = self.term("M", expansion)
            term["surfaces"][expansion] = "ci"
            term["surfaces"][acronym] = "cs"
            term["sources"].add("restored_from_composites")
            self.restored.append({"acronym": acronym, "expansion": expansion, "evidence_tokens": sorted(tokens)})

    def merge_plurals(self) -> None:
        # R2: merge a plural key into its singular only when both exist with the same type
        for tid in sorted(self.terms):
            term = self.terms.get(tid)
            if term is None:
                continue
            key = term["key"]
            for cut in ("es", "s"):
                if key.endswith(cut) and len(key) > len(cut) + 3:
                    target = f"{term['type']}:{key[: -len(cut)]}"
                    if target in self.terms and target != tid:
                        self.terms[target]["surfaces"].update(term["surfaces"])
                        self.terms[target]["sources"] |= term["sources"]
                        del self.terms[tid]
                        break

    def resolve_shared_surfaces(self) -> None:
        # A shared case-insensitive phrase means the same concept, so those terms are merged.
        # A shared case-sensitive acronym is ambiguous, so it is dropped from every owner.
        owners = collections.defaultdict(set)
        for tid, term in self.terms.items():
            for surface, mode in term["surfaces"].items():
                owners[(surface, mode)].add(tid)
        for (surface, mode), tids in owners.items():
            if mode == "cs" and len(tids) > 1:
                for tid in tids:
                    self.terms[tid]["surfaces"].pop(surface, None)
                self.remove("merge", surface, "R2", "ambiguous_acronym:" + ",".join(sorted(tids)))
        parent = {tid: tid for tid in self.terms}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (surface, mode), tids in owners.items():
            if mode == "ci" and len(tids) > 1:
                ordered = sorted(tids)
                for other in ordered[1:]:
                    ra, rb = find(ordered[0]), find(other)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        groups = collections.defaultdict(list)
        for tid in self.terms:
            groups[find(tid)].append(tid)
        merged = {}
        for members in groups.values():
            # Tasks take precedence as the primary id so that R4 applies to merged task names
            members.sort(key=lambda t: (not t.startswith("T:"), t))
            head = dict(self.terms[members[0]])
            head["surfaces"] = {}
            head["sources"] = set()
            head["types"] = set()
            head["merged_from"] = sorted(members[1:])
            for tid in members:
                head["surfaces"].update(self.terms[tid]["surfaces"])
                head["sources"] |= self.terms[tid]["sources"]
                head["types"].add(self.terms[tid]["type"])
            merged[members[0]] = head
        self.terms = {tid: t for tid, t in merged.items() if t["surfaces"]}

    def finalize(self) -> list[dict]:
        out = []
        for tid in sorted(self.terms):
            term = self.terms[tid]
            single_word_task = term["type"] == "T" and " " not in term["key"]
            # R4 also covers methods whose only surfaces are single common English words
            surfaces = term["surfaces"]
            dictionary_word_method = (
                term["type"] == "M"
                and all(m == "ci" and " " not in t and t in self.wordlist for t, m in surfaces.items())
            )
            out.append(
                {
                    "id": tid,
                    "type": term["type"],
                    "key": term["key"],
                    "surfaces": [{"text": s, "match": m} for s, m in sorted(term["surfaces"].items())],
                    "types": sorted(term.get("types", {term["type"]})),
                    "merged_from": term.get("merged_from", []),
                    "sources": sorted(term["sources"]),
                    # R4: these stay inactive until the allowlist is fixed on T1
                    "status": "r4_candidate" if (single_word_task or dictionary_word_method) else "active",
                }
            )
        return out


def main() -> None:
    study = json.loads(STUDY.read_text())
    cfg = study["vocab"]
    wordlist_path = Path(cfg["r4_wordlist"])
    wordlist = {w.strip().lower() for w in wordlist_path.read_text(errors="ignore").splitlines() if w.strip()}

    vocab = Vocab(wordlist, cfg)
    methods = pd.read_parquet(DATA / "methods.parquet", columns=["name", "full_name"])
    for name, full_name in methods.itertuples(index=False):
        vocab.add_method(name, full_name, "pwc_methods")
    # Spam overwrote some method records (for example Adam), so names attached to
    # paper annotations are used as a second source; only names are used, never counts
    tasks, annotated = set(), set()
    for path in PAPER_FILES:
        frame = pd.read_parquet(path, columns=["tasks", "methods"])
        for row in frame["tasks"]:
            if row is not None:
                tasks.update(row)
        for row in frame["methods"]:
            if row is not None:
                annotated.update((m.get("name"), m.get("full_name")) for m in row)
    for name, full_name in sorted(annotated, key=lambda x: (str(x[0]), str(x[1]))):
        vocab.add_method(name, full_name, "pwc_paper_methods")
    pairs = set(methods.itertuples(index=False, name=None)) | annotated
    vocab.restore_from_composites(pairs, cfg["restore_min_support"])
    for task in sorted(tasks):
        vocab.add_task(task)
    vocab.merge_plurals()
    vocab.resolve_shared_surfaces()
    terms = vocab.finalize()

    meta = {
        "step": "vocab_base",
        "git_commit": git_commit(),
        "rules": ["R1", "R2", "R3", "R4_candidates", "restoration"],
        "config": cfg,
        "wordlist_sha256": sha256(wordlist_path),
        "source_files_sha256": {p.name: sha256(p) for p in [DATA / "methods.parquet", *PAPER_FILES]},
        "note": "PWC paper counts (num_papers, task counts) are not used; frequency rules R5 and R6 are applied per timepoint later",
    }
    OUT_VOCAB.parent.mkdir(exist_ok=True)
    OUT_VOCAB.write_text(
        json.dumps(
            {"meta": meta, "terms": terms, "restored": vocab.restored, "removed": vocab.removed},
            indent=1,
            ensure_ascii=False,
        )
    )

    removed_by = collections.Counter((r["rule"], r["reason"].split(":")[0]) for r in vocab.removed)
    status_by = collections.Counter((t["type"], t["status"]) for t in terms)
    surfaces_by = collections.Counter(s["match"] for t in terms for s in t["surfaces"])
    result = {
        "meta": meta,
        "terms_by_type_status": {f"{a}:{b}": n for (a, b), n in sorted(status_by.items())},
        "surfaces_by_match_mode": dict(surfaces_by),
        "removed_by_rule_reason": {f"{a}:{b}": n for (a, b), n in sorted(removed_by.items())},
        "restored": len(vocab.restored),
        "merged_group_sizes": dict(sorted(collections.Counter(1 + len(t["merged_from"]) for t in terms).items())),
    }
    OUT_RESULT.parent.mkdir(exist_ok=True)
    OUT_RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved", OUT_VOCAB, "and", OUT_RESULT)


if __name__ == "__main__":
    main()