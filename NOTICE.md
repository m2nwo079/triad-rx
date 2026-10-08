# Notice

Copyright (c) 2026 m2nwo079. All rights reserved for the code and documents in
this repository, except for the third-party data and derived files listed below.

## Third-party data

| Source | Licence | Used for | Derived files in this repository |
|---|---|---|---|
| Papers with Code archive (frozen snapshot) | CC BY-SA 4.0 | Method and task vocabulary, paper-code links | `vocab/` |
| PatentsView (USPTO Open Data Portal bulk tables) | CC BY 4.0 | Patent evidence for the V axis | Patent-derived parts of `results/` and `briefings/` |
| arXiv metadata (Kaggle snapshot, Cornell University) | CC0 1.0 | Paper corpus | Paper-derived parts of `results/`, `vocab/`, `briefings/` |
| GitHub REST API, PyPI download statistics | Respective terms of service | D-axis evidence | Repository and package evidence in `results/` |

- Files under `vocab/` are adapted from the Papers with Code archive and are
  shared under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
- Patent data: attribution to PatentsView (www.patentsview.org), licensed under
  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- Generated texts in `briefings/` were produced with the Gemini API.

Raw data are not redistributed; `data/` is excluded from version control.
