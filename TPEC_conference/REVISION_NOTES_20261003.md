# Manuscript revision — October 3, 2026

Reviewed the main-paper implementation paths: continuous_eig.py, continuous_pathwise.py,
continuous_redq.py, continuous_cuda.py, continuous_rocof.py, ContinuousParticleBelief,
network construction/configuration, saved-history rescoring, and relevant validation tests.
No new experiments were launched and no simulation/training implementation was changed.

## Editorial and layout changes

- Reworked the abstract around the carried-state formulation and measured outcomes.
- Tightened methods, Bayesian updating, experimental setup, and limitations.
- Removed both unsupported allegations that test seed 1001 influenced development.
- Retained archived training seeds 101/202/303 and test seeds 1001/1002/1003.
- Removed the unverified historical explanation for duration ordering.
- Clarified that auxiliary randomness is independent of parameters, while adaptive
  actions depend on history; clarified numerical Jacobian perturbations and the
  Step-DAD simulation count including contrastive candidates.
- Clarified table uncertainty units; retained descriptive paired-comparison wording.
  No confidence interval was invented or calculated as part of this revision.
- Used normal figure placement, a compact readable duration plot, and balanced
  final-page columns. IEEEtran fonts, margins, and column widths are unchanged.
- Removed the visible larger-L placeholder and pending-work prose; the source
  retains an insertion comment. Larger-L results are not yet incorporated.

## Timing recalculation requested by the user

All costs use training seed 101. All online entries average exactly the 512
histories with evaluation seed 1001, using the sum of the saved per-stage decision
timers. No retrospective warm-up exclusion or new profiling measurements are used.
The selected rollouts, hashes, hardware, means, SDs and offline stages are recorded
in results/diagnostics/train101_eval1001_timing.json.

DAD offline = matched Fixed-stage training_seconds + DAD-stage training_seconds.
Step-DAD inherits that total. Random and Myopic have no offline training.
IEEE9 offline uses the archived A100 preparation; online uses RTX 5080 records.
IEEE14 offline and online use the archived A100 records. Tables disclose this scope;
no controlled cross-hardware comparison is claimed.

Verified DAD offline preparation hours, T=3/4/5:
IEEE9: 1.1160457653 / 1.6076185894 / 2.1391193618.
IEEE14: 3.8101691840 / 5.4566472706 / 7.4179732733.

## Verification

- 36 information-score cells match saved means/SDs at printed precision.
- 72 timing cells match the explicitly selected records at printed precision.
- DAD and Step-DAD offline totals are checked as sums of the matched stages.
- pdfLaTeX/BibTeX build: six Letter pages, no unresolved references or box warnings.
- All six rendered pages inspected; tables, figure labels, equations, author blocks
  and references are legible without clipping or overlap.
- Existing bibliography metadata was retained; this is not a fresh literature or
  affiliation verification.

Both output/eig_power_grid_conference.pdf and output/pdf/eig_power_grid_conference.pdf
are synchronized to this revision. The previous main source/PDF were backed up
outside the repository before replacement. No Git commit or push was performed.
