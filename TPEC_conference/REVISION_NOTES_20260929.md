# Minor revisions completed September 29, 2026

The current author-edited source was used as the base. Numerical table entries, experimental settings, checkpoints, and implementation were preserved. The rebuilt main PDF remains six US Letter pages. All six pages were inspected; the final LaTeX run has no unresolved references or overfull boxes.

Completed: corrected email and grammar; removed invalid apostrophe control characters; standardized IEEE 9-bus/IEEE 14-bus names; replaced the generic introduction opening with an inertia-focused sentence and a verified NREL reference; narrowed abstract and conclusion claims to measured sPCE scores; added existing numerical results to the abstract; limited online speed comparisons to Myopic and Step-DAD; removed repeated true-parameter-access statements and internal-report terminology; clarified Random's sampling distribution, offline stage-cost scope, and the time available between observations; protected Bayesian capitalization in bibliography titles. The code link already used \url{}.

## Discuss before substantial work

- Large-L evaluation and sNMC: keep saved policies and realized observations fixed, score with chunked contrastive samples, and inspect convergence and uncertainty. The existing checkpoint-reuse guard rejects contrast-count changes. Do not alter training or Step-DAD refinement contrast counts merely to change scoring.
- Increasing durations: the constraint is enforced in both training and evaluation; no physical necessity has been established. Removing it requires an explicit experiment redesign and normally retraining. Retaining it requires presenting a restricted benchmark, not inventing a physical justification.
- Paired differences: use matched evaluation seed and episode records, distinguishing uncertainty conditional on a checkpoint from training-seed variability. Three training seeds limit broad claims.
- Single-hardware timing: rerun a controlled protocol with synchronization where needed, including posterior updates and refinement, excluding scoring and plant simulation.
- Combined Fixed+DAD costs: current tables now explicitly label stage costs, but the requested combined totals remain to be calculated from matching raw per-seed records. Add times within each seed before computing SD; adding published SDs is invalid. Step-DAD inherits the same offline initialization cost.
- Results figure: choose learned durations from saved records or posterior accuracy/interval width with inference convergence checks. The conference limit is six pages including references, figures, and tables: https://tpec.engr.tamu.edu/submissions.html . Consolidate existing tables as needed.
- Myopic search-budget sensitivity and revised Random sampling require new comparisons. A uniformly sorted baseline still assumes ordered durations; an unrestricted design space calls for a different baseline.
- Scalar endpoint sensor and duration-only design require an identifiability discussion; avoid claiming that a few scalar observations identify all ten parameters.
- Broader probing, PMU-inertia-estimation, and optimal-input-design literature needs a focused source review.
- Huan--Marzouk publication request: no published version of the exact 2016 sequential paper was verified. The authors' MIT publication list labels it a preprint (https://uqgroup.mit.edu/publications/), and a 2026 SIAM article still cites the arXiv version (https://epubs.siam.org/doi/10.1137/25M1771946). Retain the accurate citation pending discussion. A distinct published 2024 review exists, but it is not the same paper.

No new experiments or retraining were started. These revisions are being committed and pushed at the user's subsequent request.

## Subsequent copyediting and larger-L feasibility pass

Completed a full manuscript copyediting pass with 39 targeted wording corrections, preserving all eight numerical tables. The rebuilt PDF remains six pages and was visually checked. Added an evaluation-only rescorer with four passing numerical tests. Thirty saved IEEE 14-bus histories were scored through L=100,000 and 18 repeated histories used for an isolated L=10,000 runtime measurement. Full-archive evaluation and retraining were not launched. See [the feasibility report](results/diagnostics/larger_L/README.md) for exact requirements, pilot records, validation, and projected runtime.
