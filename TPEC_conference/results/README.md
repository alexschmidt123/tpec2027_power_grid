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

