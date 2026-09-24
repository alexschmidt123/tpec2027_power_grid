# IEEE14 conference results

Submitted campaign complete (three evaluation seeds). The accepted conference protocol requires evaluation seeds 1001–1003; 1004/1005 are not required.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| Random | 0.7541 ± 0.0047 | 0.8360 ± 0.0297 | 0.8464 ± 0.0252 |
| Fixed | 4.2623 ± 0.0402 | 4.4946 ± 0.0345 | 4.6624 ± 0.0083 |
| Myopic | 4.2907 ± 0.0368 | 4.3990 ± 0.0177 | 4.4641 ± 0.0280 |
| DAD | 4.7576 ± 0.0362 | 4.8325 ± 0.0047 | 4.8443 ± 0.0026 |
| RL-sBOED | 4.7445 ± 0.0022 | 4.8275 ± 0.0112 | 4.8383 ± 0.0108 |
| Step-DAD | 4.7527 ± 0.0275 | 4.8297 ± 0.0067 | 4.8446 ± 0.0042 |

- [01_results](01_results/): EIG table, online timing, JSON and CSV.
- [02_configuration](02_configuration/): Protocol and resolved configurations.
- [03_models](03_models/): Checkpoints and training histories.
- [04_runs](04_runs/): Original training/evaluation outputs.
- [05_provenance](05_provenance/): Status, source map and checksum inventory.

Mean ± SD units are defined in [the full EIG table](01_results/EIG_table.md).
