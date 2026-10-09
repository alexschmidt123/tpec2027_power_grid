# Power-system result archive

| System | Main scores and model navigation |
|---|---|
| IEEE 9-bus | [ieee9/README.md](ieee9/README.md) |
| IEEE 14-bus | [ieee14/README.md](ieee14/README.md) |

The archived conference protocol uses horizons 3/4/5, training seeds 101/202/303 for fitted methods, test seeds 1001/1002/1003, 512 sequences per test seed, 128 sPCE contrasts, and four Step-DAD refinement updates. See [CONFERENCE_PROTOCOL.json](CONFERENCE_PROTOCOL.json).

Each system provides `01_results/`, `02_configuration/`, `03_models/`, `04_runs/`, and `05_provenance/`. Relative links point to preserved complete records under `experiments/`; keep this hierarchy intact. The complete rollout observations, design sequences, generating parameters, and retained checkpoints were not changed during repository cleanup.

## Manuscript timing

The authoritative manuscript costs are [diagnostics/train101_eval1001_timing.json](diagnostics/train101_eval1001_timing.json). They use training seed 101 and test seed 1001 on NVIDIA A100 PCIe 40-GB hardware, with 512 test sequences per method. DAD and Step-DAD offline preparation includes Fixed initialization and DAD training. Online decision timers exclude true-system propagation and terminal scoring; see the manuscript for the measured scope. All timing evaluations use the L=128 scoring setting.

Recalculate these values from the included records:

```bash
python3 TPEC_conference/results/diagnostics/collect_train101_eval1001_timing.py \
  --results-root TPEC_conference/results --output /tmp/tpec_timing_recalculated.json
```

The `01_results/` folders also retain historical campaign timing summaries across seeds and hardware. Those summaries are not the authoritative single-seed A100 manuscript timing table.

## Larger contrast sets

[Completed IEEE 14-bus rescoring](diagnostics/larger_L/complete_ieee14_20261005/README.md) covers 64,512 unique histories at L=10,000 and 100,000. The archive includes every history score, input/scorer hashes, the frozen scorer, and [publication_summary.json](diagnostics/larger_L/complete_ieee14_20261005/publication_summary.json).

```bash
python3 TPEC_conference/results/diagnostics/larger_L/complete_ieee14_20261005/aggregate.py
```

For fitted methods, mean ± SD summarizes three training-seed means, each averaged across all test sequences. Random and Myopic use three test-seed means. These SDs are not confidence intervals for paired method differences. Larger-L scores remain estimator-sensitive and do not certify EIG convergence.

Public metadata has machine-specific path strings removed or replaced with archive markers. Numerical values are unchanged. The complete pre-cleanup originals remain in a private archive outside the repository. Raw scientific histories and retained checkpoints are byte-identical; current public-file hashes are in [the release manifest](../../provenance/public_release_manifest.json).
