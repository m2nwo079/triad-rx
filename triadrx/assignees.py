"""Rule-based assignee classification for the V axis (design 7.5).

Categories: company, academic, government, individual, ambiguous, unknown.
- The type code keeps its last digit after removing the partial-interest prefix "1"
  (for example "12" -> "2"). An empty code or "1" alone is unknown.
- Codes 4-5 are individuals.
- Codes 2-3 and 6-9 get the same name checks: an academic pattern makes them
  academic, an academic pattern together with a company suffix makes them
  ambiguous, a company suffix makes them company, an ambiguous word without a
  company suffix makes them ambiguous. Otherwise codes 2-3 are company and codes
  6-9 government (design 7.5, v0.3.25: foreign national universities can carry a
  foreign-government code).
Word lists come from config applied.assignee_classes; matching is case-insensitive
on whole words.
"""
import re


def _word_regex(words: list[str]) -> re.Pattern:
    parts = [r"(?<!\w)" + re.escape(w) + r"(?!\w)" for w in sorted(words, key=len, reverse=True)]
    return re.compile("|".join(parts), re.IGNORECASE)


class AssigneeClassifier:
    def __init__(self, cfg: dict):
        self.academic = _word_regex(cfg["academic_patterns"])
        self.ambiguous = _word_regex(cfg["ambiguous_words"])
        self.suffix = _word_regex(cfg["company_suffixes"])
        self.prefix = cfg["partial_interest_prefix"]
        codes = cfg["codes"]
        self.org_codes = set(codes["company_or_academic"])
        self.individual_codes = set(codes["individual"])
        self.government_codes = set(codes["government"])

    def code(self, assignee_type) -> str | None:
        """Last digit after removing the partial-interest prefix; None when unknown."""
        if assignee_type is None:
            return None
        text = str(assignee_type).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if len(text) == 2 and text.startswith(self.prefix):
            text = text[1]
        if len(text) != 1 or text == self.prefix:
            return None
        return text

    def classify(self, assignee_type, organization: str | None) -> str:
        code = self.code(assignee_type)
        if code is None:
            return "unknown"
        if code in self.individual_codes:
            return "individual"
        if code not in self.org_codes and code not in self.government_codes:
            return "unknown"
        name = organization or ""
        academic = bool(self.academic.search(name))
        suffix = bool(self.suffix.search(name))
        if academic and suffix:
            return "ambiguous"
        if academic:
            return "academic"
        if code in self.government_codes:
            if suffix:
                return "company"
            if self.ambiguous.search(name):
                return "ambiguous"
            return "government"
        if self.ambiguous.search(name) and not suffix:
            return "ambiguous"
        return "company"
