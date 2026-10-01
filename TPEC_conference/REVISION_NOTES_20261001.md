# Conference framing revision

- Rewrote introduction around carried-state power-system probing and limited contributions to the benchmark.
- Added verified Pierre et al. 2010, Wall and Terzija 2014, and Hjalmarsson 2009 papers; now 11 cited references.
- Explained M and K, fixed sensor and actuator choices, duration-dependent energy, identifiability limits, and the nonphysical monotonic restriction.
- Replaced the full-width workflow diagram with an IEEE 14-bus T=5 learned-duration figure. Summary uses all 4608 histories per method across training seeds 101, 202, 303 and evaluation seeds 1001,1002,1003; quantiles use linear interpolation. Fixed variation is across fitted seeds. Summary stored in results/diagnostics/probe_durations/ieee14_T5_quantiles.json.
- Condensed Step-DAD implementation prose to retain six pages without reducing template fonts or margins.
- Existing numerical tables retained. Larger-L scoring still underway; full results, paired method differences, combined Fixed+DAD offline totals, single-hardware timings, and baseline sensitivities remain outstanding.
- Compiled with pdflatex/bibtex and inspected all six rendered pages; no unresolved references or overfull boxes.
- Overleaf has not been synchronized.

Verified source records:
https://data.pnnl.gov/group/nodes/publication/15375
https://research.manchester.ac.uk/en/publications/simultaneous-estimation-of-the-time-of-disturbance-and-inertia-in/
https://www.sciencedirect.com/science/article/pii/S0947358009709902
