# TPEC result index

| Benchmark | Role | Saved evaluation seeds | Status | Open |
|---|---|---|---|---|
| SIR-ODE | Supplementary benchmark | 1001–1005 | Complete | [Results](sir_ode/README.md) |
| IEEE9 | Main power-grid case | 1001–1003 | Complete (accepted three-seed protocol) | [Results](ieee9/README.md) |
| IEEE14 | Main power-grid case | 1001–1003 | Complete (accepted three-seed protocol) | [Results](ieee14/README.md) |

Each benchmark has the same five sections:

1. `01_results/` — tables and numeric summaries.
2. `02_configuration/` — experiment settings.
3. `03_models/` — saved models and training histories.
4. `04_runs/` — original evaluations and training runs.
5. `05_provenance/` — completeness, source locations and verification.

Methods use the same row order: Random, Fixed, Myopic, DAD, RL-sBOED, Step-DAD; columns are T=3,4,5. Unavailable groups are Pending, not zero.

The benchmark folders provide links into preserved files. Grid raw files are fully collected under `experiments/`. SIR originals remain in `../sir ode result/`. All navigation links resolve within TPEC_conference; preserve this hierarchy when copying the folder.

SIR reports finite-particle entropy reduction (ceiling ln(1024)); grids report sPCE with 128 contrasts (ceiling ln(129)). SIR's historical timing scope includes observation lookup and posterior updating and differs from grid decision-only timing. Do not pool these metrics or claim cross-hardware speedups.

See `STATUS.json`, `FINAL_VERIFICATION.json`, and `collector_status.json` for completion and verification. Collection has finished; no ongoing SSH session is required to use these results. Evaluation seeds 1004/1005 are outside the accepted grid conference scope. See `CONFERENCE_PROTOCOL.json` for the final scope.


## Completed follow-up results (October 5, 2026)

The completed conference evaluations, saved models, training histories, configurations, and numerical summaries are available in this repository. Additional HPRC submission records, source manifests, and logs are preserved under `experiments/`; the transfer receipt is in `diagnostics/hprc_transfer_20261003/`.

IEEE9 A100 timing for training seed 101/test seed 1001 is complete (512 test episodes per method; Grace job 19952383). Raw results are in `diagnostics/a100_online_profile_test1001_20261004/`. Both manuscript cost tables now use A100 measurements from `diagnostics/train101_eval1001_timing.json`; the adjacent collector reproduces these values. DAD and Step-DAD preparation includes Fixed and DAD training. IEEE14 larger-L rescoring is complete: all 64,512 unique histories at L=10,000 and 100,000. Raw records, frozen scorer, input checksums, portable aggregation, and seed-level summaries are in `diagnostics/larger_L/complete_ieee14_20261005/`. The manuscript includes the complete mean-score table and discusses estimator-sensitive rankings and residual contrast-count sensitivity.

See `diagnostics/publication_audit_20261003.json` and `diagnostics/final_manuscript_verification_20261005.json` for the completed publication audit and numerical verification. No planned conference result remains pending.
