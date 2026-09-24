# tpec2027_power_grid

Conference code, paper, and complete archived EIG results for informative power-system probing with sequential Bayesian optimal experimental design. The study applies existing design methods to acquire more information about uncertain inertia and damping parameters under the same number of design steps.

**Methods:** Random, Fixed, Myopic, DAD, RL-sBOED, and Step-DAD. IEEE9 and IEEE14 are the main benchmarks; SIR-ODE is supplementary. This repository contains no MoE-sBOED, MOCU, or MSC implementations. It has an independent Git history.

## Paper and complete results

- [Conference paper (PDF)](TPEC_conference/output/pdf/eig_power_grid_conference.pdf), [LaTeX](TPEC_conference/eig_power_grid_conference.tex), and [independent bibliography](TPEC_conference/conference_refs.bib).
- [Result index](TPEC_conference/results/README.md): [IEEE9](TPEC_conference/results/ieee9/README.md), [IEEE14](TPEC_conference/results/ieee14/README.md), [SIR-ODE](TPEC_conference/results/sir_ode/README.md).
- [Accepted conference protocol](TPEC_conference/results/CONFERENCE_PROTOCOL.json), [grid model index](TPEC_conference/results/MODEL_INDEX.json), and [release verification](provenance/release_verification.json).

All three benchmarks cover **T=3,4,5** and training seeds **101,202,303**. Grid evaluations use **1001–1003** (512 episodes per seed); SIR evaluations use **1001–1005**. Random and Myopic do not have independent training replications. Step-DAD reuses its matching DAD model. Fixed has a learned grid sequence or a saved SIR calibration sequence.

The archive includes raw evaluations, per-system trajectories, aggregate tables, resolved configurations, training histories, timing records, checkpoint provenance, **72 indexed model files (54 grid and 18 SIR)**, and additional archived checkpoint copies (**82 `.pth` files in total**). Result and checkpoint bytes are unchanged. Files are stored directly in Git; no separate model download or Git LFS is required. The complete checkout is approximately 0.8 GB before Git overhead.

The grid EIG values use sPCE with **128 contrasts**, giving a ceiling of `log(129)`. New runs default to **1024 contrasts**; explicitly use `--contrasts 128` to match the archived conference protocol. SIR uses finite-particle posterior entropy reduction on 1024 support particles, a different estimator. Timing scopes also differ; see the result index before making comparisons.

## Installation

Use Linux, Python 3.10+, an NVIDIA GPU and a matching CUDA toolkit containing `nvcc`. Install the appropriate CUDA-enabled PyTorch build for your GPU, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the environment before running; `nvcc` must be on `PATH`. The launchers use the active `python3`, or an explicit `PYTHON=/path/to/python`. Grid simulation uses PyCUDA; SIR supports PyTorch CPU or CUDA. Do not install or expose machine-specific account credentials.

## Run

The root launchers are the entry points for complete experiments. Each invocation creates a fresh timestamped run directory and trains only the requested methods (Step-DAD also requires DAD).

```bash
# Tiny workability checks; these are not performance results.
bash run.sh --config configs/ieee9_eig.yaml --smoke
bash run.sh --config configs/ieee14_eig.yaml --smoke
bash run.sh --config configs/sir_ode_eig.yaml --eval-seeds 1001 --smoke

# One grid conference cell; T and training seed must be varied for full replication.
bash run.sh --config configs/ieee14_eig.yaml --T 3 --seed 101 \
  --eval-seeds 1001,1002,1003 --updates 2000 --batch-size 32 \
  --learning-rate 0.001 --validation-systems 128 --validate-every 100 \
  --eval-systems 512 --contrasts 128 --planner-particles 128 \
  --search-candidates 16 --search-rounds 3 --fantasies 32 \
  --refinement-updates 4 --window 3.5 --observation-kind endpoint_rocof

# Optional sweep: starts fresh training for every cell.
bash sweep_run.sh --configs ieee9_eig,ieee14_eig --T 3,4,5 \
  --seed 101,202,303 --eval-seeds 1001,1002,1003 \
  --contrasts 128 --eval-systems 512
```

Use `--methods random,fixed,myopic,dad` to select a subset. Grid probes have fixed amplitude 0.05 pu at bus 1, strictly increasing durations in [0.2,3] s with a 0.01 s gap, 3.5 s observation windows, signed endpoint RoCoF, and full state carry between probes. Methods are documented adaptations, not exact reproductions of their source papers; the SIR and grid training recipes differ.

SIR generates its deterministic ODE bank automatically into `data/sir_ode/`; smoke tests use a separate bank inside their own output directory. Full banks are regenerated from the recorded seeds and configuration and are not part of the result archive. Inspect the archived resolved configuration when reproducing a historical run.

## Verification and provenance

```bash
python tools/verify_release.py
python -m unittest discover -s tools/tests -v
```

The first command checks copied-file SHA-256 hashes, portable navigation, excluded implementation names, and strict loading of every checkpoint into the retained architectures. Numerical tests cover ordered duration feasibility, state carry, EIG gradients and REDQ behavior.

[Migration manifest](provenance/migration.json) records the original source revision and every copied file. Historical mixed-objective source snapshots and workstation collection tools were omitted from this public repository to keep its code within conference scope; their hashes and omission reasons are recorded, and originals remain in the source project. Saved result JSON, configuration snapshots and model metadata may still mention inactive historical options; they are preserved as provenance, not executable implementations. Historical manifests describe the original collection, while `provenance/release_verification.json` describes this release.

The model loader accepts all archived checkpoints. `--evaluate-from` is intentionally stricter: it requires a completed grid run produced by this repository with matching scientific source and settings. It will reject historical mixed-project snapshots; this prevents silently presenting a refactor as an exact archived source match. Neither training resume nor automatic checkpoint reuse is performed.
