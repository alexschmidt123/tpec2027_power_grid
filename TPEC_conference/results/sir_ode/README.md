# SIR-ODE supplementary results

Complete: three training seeds × five evaluation seeds at T=3,4,5; 512 episodes per evaluation.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| Random | 3.7783 ± 0.0167 † | 3.8593 ± 0.0167 † | 3.9241 ± 0.0167 † |
| Fixed | 6.2816 ± 0.0101 † | 6.4501 ± 0.0091 † | 6.5360 ± 0.0044 † |
| Myopic | 6.7108 ± 0.0125 † | 6.7437 ± 0.0124 † | 6.7596 ± 0.0135 † |
| DAD | 6.8394 ± 0.0087 | 6.8591 ± 0.0039 | 6.8658 ± 0.0077 |
| RL-sBOED | 6.8494 ± 0.0029 | 6.8674 ± 0.0020 | 6.8814 ± 0.0015 |
| Step-DAD | 6.8438 ± 0.0095 | 6.8679 ± 0.0011 | 6.8761 ± 0.0109 |

DAD/RL-sBOED/Step-DAD: mean ± sample SD across three training-seed means, each averaged over five evaluations.
† Myopic/Fixed/Random: outputs are identical across training seeds for each evaluation seed; duplicate copies are counted once. Their ± is SD across five evaluation-seed means, not training variability.
No workspace duplicates are counted. These are 3 independent trainings, not 15.

Metric: finite-particle posterior entropy reduction with 1,024 particles; uniform-prior entropy ceiling ln(1024) ≈ 6.9315 nats. It is not grid sPCE.


- [01_results](01_results/)
- [02_configuration](02_configuration/)
- [03_models](03_models/)
- [04_runs](04_runs/)
- [05_provenance](05_provenance/)

Original verified package: ../../sir ode result/. Grid and SIR information estimators differ; their absolute scores and runtime scopes must not be pooled.
