# TPEC conference paper and results

The six-page IEEE conference source is [eig_power_grid_conference.tex](eig_power_grid_conference.tex), with its independent [conference_refs.bib](conference_refs.bib).
Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The source uses IEEEtran conference mode with standard margins and fonts. Three author placeholders are provided; remove the third block and its preceding `\and` for two authors.

## Paper organization

The main paper is six pages in standard IEEEtran conference format, including references. It preserves the default margins, two columns, and font sizes. The final-page columns are balanced without stretching text to fill the page.

- The abstract summarizes the motivation, approach, and findings. The introduction explains frequency dynamics, uncertain M/K, probing experiments, Bayesian design, and expected information gain before stating the research questions.
- The formulation specifies the reduced networks, carried physical state, pulse-duration constraints, noisy endpoint RoCoF observations, and sequential information objective. Figure 1 appears on page 2 and distinguishes history-based decisions from explicit particle-posterior planning.
- The method comparison separates observation feedback, optimization horizon, and offline versus online computation. These are adaptations of existing algorithms, not newly proposed algorithms.
- Experimental settings distinguish independent training seeds, repeated evaluations, checkpoint selection, and hardware used for timing.
- Results contain eight tables: physical parameters, method properties, two information-score tables, and four timing tables. Numerical entries are unchanged by the editorial revision. Grid information scores estimate the sPCE lower bound on EIG; they are not exact EIG.
- Discussion distinguishes information acquisition from demonstrated stability or control benefits, and retains initialization-cost, estimator-saturation, development-data, and zero-latency limitations.

The unrelated historical SIR-ODE appendix is preserved as a separate [supplement](sir_ode_supplement.tex), with a [compiled PDF](output/pdf/sir_ode_supplement.pdf). Its numerical results and methodological qualifications are unchanged. Build it with pdfLaTeX twice. It does not count toward the six-page main paper or support its power-grid conclusions.

Author names, affiliations, and email addresses remain placeholders and must be supplied before submission.

## Result collection on LabPC

Open [the result index](results/README.md), then select:

- [SIR-ODE](results/sir_ode/README.md)
- [IEEE9](results/ieee9/README.md)
- [IEEE14](results/ieee14/README.md)

Each benchmark has `01_results`, `02_configuration`, `03_models`, `04_runs`, and `05_provenance`. These views link to preserved, verified data. Original grid campaign paths under `results/experiments` and the original `sir ode result` package remain unchanged. Collection is complete and all navigation links resolve within TPEC_conference. No live SSH session is required to use the collection. The verification report is `results/FINAL_VERIFICATION.json`; the accepted scope is `results/CONFERENCE_PROTOCOL.json`.

This public repository includes the complete conference results, saved models and rendered paper. Tables and the workflow diagram are embedded in the LaTeX. The PDF is `output/pdf/eig_power_grid_conference.pdf`. The broader project manuscript is outside this conference repository.

The complete result collection includes 54 grid and 18 SIR saved model files, raw evaluations, configurations, training histories, saved configuration snapshots, source-hash records, checksum records, and aggregate EIG/online/offline summaries. IEEE9 Step-DAD T=5 is 2.6621 ± 0.0145 nats across three training-seed means. The six-page main paper incorporates the completed grid results; historical SIR results remain in the separate supplement.

The paper code link points to https://github.com/alexschmidt123/tpec2027_power_grid. Historical mixed-objective executable snapshots were omitted; see `../provenance/migration.json` and the root README. Raw numerical records are unchanged.
