"""Count arXiv ML papers per year."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CATEGORIES = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "stat.ML"]
OUT = Path("data/scope/arxiv_counts.json")
YEARS = range(2014, 2027)


def count(year: int) -> int | None:
    # The 2026 window stops at the snapshot date
    end = f"{year}12312359" if year < 2026 else "202609302359"
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    query = f"({cats}) AND submittedDate:[{year}01010000 TO {end}]"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": 1}
    )
    for attempt in range(3):
        try:
            text = urllib.request.urlopen(url, timeout=60).read().decode()
            return int(re.search(r"<opensearch:totalResults[^>]*>(\d+)<", text).group(1))
        except Exception as exc:
            # Back off longer on each failure: 5s, 10s, 20s
            wait = 5 * 2**attempt
            print(f"  {year} attempt {attempt + 1} failed: {exc!r}, retry in {wait}s", flush=True)
            time.sleep(wait)
    return None


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Resume from a previous run and only fetch missing years
    counts = json.loads(OUT.read_text()) if OUT.exists() else {}
    for year in YEARS:
        key = str(year)
        if counts.get(key) is not None:
            print(year, counts[key], "(cached)")
            continue
        counts[key] = count(year)
        print(year, counts[key], flush=True)
        # arXiv API asks for at least 3 seconds between calls
        time.sleep(3.5)
    OUT.write_text(json.dumps(counts, indent=2))
    missing = [y for y, n in counts.items() if n is None]
    if missing:
        sys.exit(f"Missing years after retries: {missing}. Run the script again.")


if __name__ == "__main__":
    main()