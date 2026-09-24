# SIR-ODE · TPEC conference results

**Six methods · T=3/4/5 · 3 training seeds × 5 evaluation seeds**

Merged LabPC and HPRC results. All 45 evaluation sets and saved model files are included and verified.

## 1. Main result

Mean EIG in nats; higher is better.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| DAD | 6.8394 | 6.8591 | 6.8658 |
| RL-sBOED | 6.8494 | 6.8674 | 6.8814 |
| Step-DAD | 6.8438 | 6.8679 | 6.8761 |
| Myopic | 6.7108 | 6.7437 | 6.7596 |
| Fixed | 6.2816 | 6.4501 | 6.5360 |
| Random | 3.7783 | 3.8593 | 3.9241 |

[Full table with standard deviations](01_results/EIG_table.md) · [All evaluation summaries (CSV)](01_results/merged_evaluation_summary.csv)

The three DAD-family methods have similar mean EIG and lie near the finite-particle entropy ceiling (6.9315 nats). Small mean differences alone do not establish superiority.

## 2. Experiment at a glance

| Setting | Value |
|---|---|
| Benchmark | SIR ordinary differential equation model |
| Observation | Infected count; one scalar; noise σ = 1.0 |
| Design | Increasing measurement times; 100 candidate times |
| Probes per experiment | 3, 4 or 5 |
| Training seeds | 101, 202, 303 |
| Evaluation seeds | 1001, 1002, 1003, 1004, 1005 |
| Systems per evaluation | 512 |
| Posterior-support particles | 1,024 |
| Training budget | 200 epochs; 512 steps/epoch; batch 256 |
| Learning rate / hidden width | 0.0003 / 256 |
| Validation systems | 256 |

[Detailed settings and historical method adaptations](02_configuration/CONFIGURATION.md)

## 3. Find what you need

| Folder | Contents |
|---|---|
| [01_results](01_results/README.md) | EIG tables, per-seed summaries and numeric aggregates |
| [02_configuration](02_configuration/CONFIGURATION.md) | Readable settings and all 45 original resolved configurations |
| [03_models](03_models/README.md) | DAD/RL checkpoints, Fixed designs, training histories and model index |
| [04_runs](04_runs/README.md) | Original evaluations, per-system rollouts, logs and configurations |
| [05_provenance](05_provenance/README.md) | Source locations, checksums and verification reports |

## 4. How to read uncertainty

- DAD, RL-sBOED and Step-DAD: SD across three training-seed means; each mean uses five evaluation seeds.
- Myopic, Fixed and Random: duplicate results across training seeds are counted once. Their marked SD is across five evaluation-seed means.
- These are three independent trainings per horizon, not fifteen.

## 5. Model files

DAD and RL-sBOED have `.pth` checkpoints. Step-DAD starts from its matching DAD checkpoint. Fixed has a saved design JSON. Myopic and Random have no trained model.

This package preserves existing results without retraining. It does not include the simulation bank or a complete historical software environment. Original experiments and currently running jobs were not changed.
