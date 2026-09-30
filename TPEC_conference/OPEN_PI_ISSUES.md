# Outstanding PI feedback after the minor revision

Status: September 29, 2026. Minor editorial corrections have been applied; experiments and numerical table entries remain unchanged. Significant or disputed changes await discussion, following the PI's instruction.

## Must-fix technical issues

### 1. IEEE 14-bus estimator saturation — pilot completed; production reevaluation unresolved
Evaluation still uses L=128 and has ceiling log(129)=4.8598 nats. At T=4/5, learned-policy scores lie around 4.83–4.84, so the comparison cannot resolve their true EIG ranking. A separate archived-history sPCE/sNMC scorer is now implemented and tested. Thirty diagnostic histories were scored through L=100,000, plus an isolated runtime check at L=10,000. Full-archive reevaluation at L=100,000, with nested L=10,000 scores, was launched September 30 on LabPC and is in progress; final results are not yet available. See [the larger-L report](results/diagnostics/larger_L/README.md). The current single-GPU estimates including checks are about 28 hours at L=10,000 and 124 hours at L=100,000; convergence is not guaranteed by either choice. Rescore fixed saved histories at 10,000 contrastive samples, check convergence toward 100,000 where needed, report both bounds and Monte Carlo uncertainty. Use chunked likelihood computation and carried states. Bounds concern expectations; a finite empirical lower/upper pair is not itself a confidence interval for EIG. Keep scoring contrast count separate from training, Myopic planning, and Step-DAD refinement. The current checkpoint-reuse guard rejects a changed contrast count.

### 2. Increasing pulse durations — unresolved
All methods enforce D_t-D_(t-1)>=0.01 s, with D_t in [0.2,3] s and an upper bound reserving space for future probes. No physical necessity for ordering has been established. Do not describe this as an operational requirement or assume its origin has been proven. Decide whether to retain it as a restricted benchmark or remove it consistently from all implementations and retrain affected policies. Simply deleting the equation or remapping existing policies would not support an unrestricted-policy optimum claim.

### 3. Paired statistical comparisons — unresolved
Tables still report seed-level means and SDs. Step-DAD's IEEE 9-bus gains over DAD are 0.0105, 0.0287, and approximately 0.0269 nats; no paired confidence intervals are available. Match evaluation seed, episode, and corresponding training seed/checkpoint when computing differences. Distinguish uncertainty over evaluation episodes from variability across independently trained policies. Do not treat repeated episodes or identical Random/Myopic evaluations as independent training replicates. Three training seeds remain a limitation. SD overlap alone neither establishes nor refutes a paired difference.

### 4. Controlled online timing — unresolved
The timing data remain from RTX 5080/A100 runs, with RTX 4090 used in development and one IEEE 9-bus Step-DAD T=5 training-seed-303 evaluation on A100. Rerun all six methods on one machine under a documented, consistent protocol. Include appropriate device synchronization and warm-up; keep batch size, measurement scope, and episode workload explicit. Include posterior updates/refinement, excluding plant simulation and score computation. Existing abstract timing numbers are recorded results, not new controlled measurements.

### 5. Total offline cost — partially addressed
Tables V/VI are now labeled offline stage time, and text explicitly states that DAD excludes Fixed initialization. The requested combined Fixed+DAD entries have not been calculated. Sum matched raw per-training-seed times, then compute their mean and sample SD; do not add SDs. Step-DAD inherits the corresponding combined offline cost. Retain clarity about hardware differences and shared initialization accounting.

## Should-fix substance

### 6. Parameter accuracy/uncertainty results — unresolved
No posterior error, credible-interval width, or coverage results for M and K have been added. Compare methods using the same inference procedure and validate its accuracy/convergence, particularly for the ten-parameter IEEE 14-bus posterior. Choose scale-normalized errors or clearly stated physical units. Interval width alone does not establish accuracy: assess coverage or calibration where feasible. Avoid evaluating only methods that already maintain online particle posteriors, since that would confound policy and inference quality.

### 7. Learned probe-duration figure — unresolved
The main paper still has a workflow figure and eight tables, but no experimental-results figure. Saved episode records contain duration sequences and observations. Plot Fixed and learned policies by stage, showing variability and, where informative, dependence on early measurements. Explain behavior under carried dynamics without attributing mechanisms unsupported by the plot. Consolidate tables to fit the six-page limit.

### 8. Myopic budget adequacy — unresolved
Myopic still uses 16 proposals, three rounds, and 32 simulated observations per proposal, with 128 posterior particles. No search-budget sensitivity or convergence evidence has been added. Test larger budgets on matched episodes and report information/cost tradeoffs. Until then, Fixed beating this implementation does not establish a general weakness of Myopic optimization.

### 9. Random sequence distribution — partially addressed
Text now accurately says sequential uniform draws are not uniform over feasible full sequences. The sampler and scores remain unchanged, including the low IEEE 14-bus score near 0.75 at T=3. If ordered durations remain, implement a uniform distribution over the feasible ordered sequence region with the gap constraint and compare on paired draws. If ordering is removed, choose a matching unrestricted random baseline. Uniform sorted draws without accounting for the minimum gap are not automatically the exact desired constrained distribution. Check whether low scores result from sequence bias, sensor sensitivity, or another implementation issue rather than assuming a bug.

### 10. Scope and identifiability — unresolved
Only duration is optimized; amplitude remains 0.05 pu, injection and sensing are at bus 1, and the endpoint RoCoF observation is at 3.5 s after the pulse ends. There are only 3–5 scalar observations for ten IEEE 14-bus parameters. The paper lacks a substantive rationale for these choices and an identifiability discussion. Explain the controlled scope and limitations; examine parameter sensitivities/posterior contraction where feasible. Fewer observations than parameters does not by itself invalidate Bayesian uncertainty reduction, but individual parameter identification must not be implied without evidence.

### 11. Power-system context and related work — partially addressed
The introduction now opens with declining physical inertia and cites a verified NREL report. IEEE 14-bus inertia values are explicitly assumed benchmark values, which satisfies the PI's allowed wording alternative without changing the model. Broader BPA/PNNL probing, PMU-based inertia estimation, and optimal input-design literature has not been added. Connect those works to the specific contribution and explain the reduced benchmark's relationship to operational grid identification. Do not retrospectively attribute assumed values to a dataset they were not taken from.

## Writing and conference presentation

### 12. Conference argument and structure — partially addressed
Internal-report terms and repeated parameter-access statements have been reduced. The paper still has eight tables, lengthy algorithm/training descriptions, tutorial material, and no results figure. It needs a clearer research question, focused contribution statement, condensed methods, and results organized around findings with physical interpretation. Detailed architectures, checkpoint bookkeeping, initialization choices, and reproducibility records can remain in the repository. Preserve the author's voice and distinguish evidence from prospective benefits.

### 13. Broad conventional-probing claim — wording corrected
The copyediting pass removed the claim that all conventional probing is fixed or myopic and replaced the dynamic-changes wording with the precise limitation that Fixed does not use incoming measurements. These experiments do not claim to demonstrate adaptation to parameter drift.

### 14. Huan–Marzouk published-version request — disputed/unverified
Bayesian capitalization was fixed in the bibliography. No published version of the exact 2016 sequential approximate-dynamic-programming paper was verified. The authors' MIT list labels it a preprint, and a 2026 SIAM article still cites arXiv. Retain the accurate reference pending discussion; a separate 2024 published review is not a published version of this paper. Sources: https://uqgroup.mit.edu/publications/ ; https://epubs.siam.org/doi/10.1137/25M1771946 .

### 15. Page space after substantive revision — pending
The present revised PDF compiles to six US Letter pages and was visually checked. TPEC requires six pages including figures, tables, and references: https://tpec.engr.tamu.edu/submissions.html . This check must be repeated after new results and references are added; table consolidation and text cuts are needed to make room.

## PI items completed in the minor pass and subsequent copyediting

Corrected tamu.edu email, serves/integrate wording, contrastive-sample terminology, nats terminology, sampled-score upper-bound wording, ambiguous deadline phrase, network naming, and invalid apostrophe characters. Reduced repeated access-to-true-parameter statements and internal-report language. Corrected the abstract speed comparison to Myopic/Step-DAD and added existing numerical findings. The GitHub link already used \url{}. Clearly described IEEE 14-bus inertia as assumed benchmark values. Rebuilt the six-page PDF with no unresolved references or overfull boxes; numerical table entries were preserved.

The subsequent full manuscript copyediting pass applied 39 targeted corrections, including the abstract, Bayesian-update explanation, seed-aggregation wording, timing description, and conclusion. Numerical tables are unchanged. The rebuilt PDF still has six pages, with all pages visually checked and no unresolved references or overfull boxes. The broader conference-structure revision is still pending.
