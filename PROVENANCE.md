# Provenance

This repository is a condensed public history of TriadRx up to the fifth study
(matching precision and the operational definition of combination). Every commit is a snapshot of a boundary commit in the full
archived history (https://github.com/m2nwo079/triad-rx-archive, private). Intermediate commits were folded.

Preserved as separate commits so that frozen documents can be checked with
`git show <freeze commit>:<file>`:

- Preregistration freeze and its record commit (first study)
- Prospective registration of the top 100 prescriptions
- Engine preregistration freeze and its record commit (second study)
- Matching preregistration freeze and its record commit (third study)
- Represcription preregistration freeze, its record commit and the prospective registration (fourth study)
- Faithfulness preregistration freeze and its record commit (fifth study)

Differences from the archived snapshots:

- The `고정 커밋:` lines in `preregistration.md`, `preregistration_engine.md` and
  `preregistration_matching.md` point to the new freeze commits.
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
| 15f04ef | cb0f419 | feat(matching): refine matching, review common words and synonyms, and rebuild vocabularies |
| f6fc015 | c5e9bc8 | feat(matching): label v3 with the relation classifier, first confirmation and T1, R2, R3 labels |
| 95bec80 | 160a90b | data(matching): adopt label v3 after the second confirmation |
| 0a3ae39 | 19d387f | feat(matching): rebuild the engine on v3 inputs and diagnose on T2 |
| b72ca7f | d6a3453 | feat(matching): fix the T3 verdict rules, power analysis and T3 code |
| b01783e | cd89808 | docs(prereg): freeze the matching preregistration |
| e2b99c0 | 1655871 | docs(prereg): record the matching preregistration freeze commit |
| 835d8b6 | a373326 | data(matching): classify the T3 label window and add the T3 verdict |
| eddc55a | d1b8d5e | docs(report): add the third-study section and bump the design version to v0.5.2 |
| 5c8476e | 136da36 | feat(represcription): rebuild the applied window with v3 rules and review new terms |
| ca5c72d | 9510e8e | feat(represcription): add applied engine inputs and the locked prescription ranking |
| 4b924e2 | b1717ad | docs(prereg): freeze the represcription preregistration |
| 21ccc54 | 4b47349 | docs(prereg): record the represcription preregistration freeze commit |
| 40505d6 | a1e6d76 | data(prospective): register the represcription top 100 and its baseline |
| 729eedb | 78ca1e8 | data(represcription): collect evidence, compute DVF and generate briefings |
| c884609 | f36e526 | data(represcription): score Q1, compare with the first study and report the fourth study |
| 3082945 | e47dbfc | feat(faithfulness): add prompts v4, the business basis and the study code |
| 546e0e4 | c32b5fc | docs(prereg): freeze the faithfulness preregistration |
| ea790cf | 1003d7d | docs(prereg): record the faithfulness preregistration freeze commit |
| 32af1a0 | d421a63 | data(faithfulness): generate with prompts v4 and score Q1 |
| 330f3e6 | 656f1cd | data(faithfulness): add the blind re-judging and report the fifth study |
| 29ba403 | 0667325 | docs(figures): add paper figures built from result files |

