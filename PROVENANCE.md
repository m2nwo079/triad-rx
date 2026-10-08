# Provenance

This repository is a condensed public history of TriadRx up to the second study
(engine reliability). Every commit is a snapshot of a boundary commit in the full
archived history (https://github.com/m2nwo079/triad-rx-archive, private). Intermediate commits were folded.

Preserved as separate commits so that frozen documents can be checked with
`git show <freeze commit>:<file>`:

- Preregistration freeze and its record commit (first study)
- Prospective registration of the top 100 prescriptions
- Engine preregistration freeze and its record commit (second study)

Differences from the archived snapshots:

- The `고정 커밋:` lines in `preregistration.md` and `preregistration_engine.md`
  point to the new freeze commits.
- `CLAUDE.md` (working notes) and `external/tod/` (results of a private earlier
  project) are not included. The tod comparison in `results/rl_case.json` and the
  reports is kept, but `scripts/briefing/03_rl_case.py` cannot be rerun here.
- The Kaggle username in `kaggle/step4/kernel-metadata.json` is a placeholder.

All other commit hashes in result files, change logs and reports refer to the
archived history. The order of individual preregistration steps (criteria
committed before each sample was drawn) is recorded only in the archive.

| Archived commit | Commit here | Content |
|---|---|---|
| 7819f89 | 4039a83 | chore: scaffold project and add stage 1 scope survey |
| 00f8f94 | b6c3703 | feat(collect): add arXiv snapshot filter with data cutoff |
| 109bffa | 56ca888 | feat(vocab): build vocabulary with rules R1-R6 and M1 measurement |
| 783beac | 6e9f1eb | feat(model): add hypergraph, triad candidates, labels, features and model |
| 1c4e10d | 5cec851 | docs(prereg): freeze preregistration before opening T2 labels |
| 4b7eeea | a3db782 | docs(prereg): record preregistration freeze commit |
| 19bc8ce | 23f74eb | feat(validation): evaluate T2 and rank applied-mode candidates |
| 443c23d | 7167454 | data(prospective): register top 100 prescriptions |
| 92d2770 | d2f7a5d | feat(evidence): collect evidence and compute DVF scores |
| 015cc52 | 5702577 | feat(briefing): generate briefings and run quality checks |
| 066fccd | 2317625 | feat(report): generate reports and lock prospective verification |
| 4ad524d | 7d41965 | feat(engine): develop engine reliability study |
| eabee61 | 13b2683 | docs(prereg): freeze the engine preregistration |
| ba63525 | 953320c | docs(prereg): record the engine freeze commit |
| 4c9f60f | b135ccf | feat(engine): add T3 verdict and close the engine study |

