# TPEC manuscript

The professor-reviewed source is [eig_power_grid_conference.tex](eig_power_grid_conference.tex), with [conference_refs.bib](conference_refs.bib). Repository cleanup did not change either file. Tables and plots are embedded in the source.

Run `bash TPEC_conference/build.sh` from the repository root, with a TeX distribution providing IEEEtran, BibTeX, TikZ, and PGFPlots. Intermediate files go into ignored `build/`; the sole public manuscript PDF is [output/pdf/eig_power_grid_conference.pdf](output/pdf/eig_power_grid_conference.pdf).

The current manuscript has six pages, uses IEEEtran conference mode on US Letter, and includes the references. [SUBMISSION_CHECK.md](SUBMISSION_CHECK.md) records the format check and the remaining copyright/PDF eXpress actions. Author blocks contain named authors rather than placeholders.

See [results/README.md](results/README.md) for the information-score data, A100 timing records, complete IEEE 14-bus larger-contrast rescoring, configurations, and checkpoints. Historical SIR supplements and internal editorial discussions are not part of this public manuscript package.
