# Completed IEEE 14-bus archived-history rescoring

Completed October 5, 2026: 64,512 unique histories, six methods, T=3/4/5, training seeds 101/202/303 where applicable, test seeds 1001/1002/1003, and 512 episodes per test seed. Policies, durations, observations, and generating parameters are unchanged. L=10,000 is a prefix of L=100,000; matched evaluation-seed/episode pairs share prior contrast streams.

`results/scores.jsonl` contains every per-history result. `results/manifest.json` records input and scorer checksums; `source/tools/rescore_saved_histories.py` is the frozen scorer. `publication_summary.json` contains manuscript means, sample SDs of seed means, and seed-level summaries. For fitted methods, each training-seed mean averages all test episodes; Random/Myopic SDs use test-seed means. Original states and L=128 scores were reproduced before rescoring.

Reproduce the aggregation without a GPU:

```bash
python3 aggregate.py
```

The full physics rescoring requires the exact original simulator snapshot specified in the manifest; aggregation needs only the included archive. The rescoring runtime is separate from the manuscript's A100 training/decision costs.

Myopic exceeds Fixed at both larger contrast counts at every horizon. Learned-policy means remain above the baselines. Step-DAD has no consistent mean improvement on IEEE14. Scores remain sensitive to L; these results do not certify EIG convergence or fine learned-policy rankings. Empirical sNMC/sPCE gaps in the archive are not confidence intervals.
