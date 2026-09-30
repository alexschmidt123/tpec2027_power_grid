# IEEE 14-bus larger-L feasibility and implementation report

September 29, 2026. This is an evaluation pilot and runtime study, not a replacement for the conference results. No policy was retrained or rerun, and no experimental table was changed.

## Production run started September 30, 2026

Full archived IEEE 14-bus rescoring is now running on LabPC's RTX 4090 at L=100,000, retaining nested L=10,000 results as a convergence diagnostic. Scope: 64,512 histories, all six methods, T=3/4/5, training seeds 101/202/303 where applicable, and evaluation seeds 1001/1002/1003. No policy retraining, new policy rollouts, or constraint changes.

Results and logs: `experiments/20260930_ieee14_L100000/results/` and `experiments/20260930_ieee14_L100000/run.log`. The launched scorer is frozen under that job's `source/tools/` directory. `launch.json` stores the PID and exact launch/resume commands. `results/progress.json` updates after the first history and every ten histories; `results/scores.jsonl` contains each completed history. Completion requires `summary.json` and progress status `complete`.

Restart support is implemented and tested: `--resume` requires matching code, input checksums, and settings, validates saved records, and skips completed identities. Five unit tests pass; a six-history GPU integration run resumed without adding duplicate rows. The original roughly five-day runtime estimate remains preliminary. Initial production histories reproduced archived scores/states and were processed successfully. Final updated paper results are pending; L=100,000 does not itself certify convergence.

The remainder records the September 29 pilot; statements below about a full run not yet being started describe that earlier state.

## Completed

- Finished a line-by-line copyediting pass on the author-edited manuscript, with 39 targeted corrections to grammar, wording, and unnecessary repetition. Rebuilt and inspected all six PDF pages; the final build has no unresolved references or overfull boxes. All eight numerical tables are unchanged.
- Implemented `tools/rescore_saved_histories.py` to score archived observations and durations under the original prior using the original carried-state simulator.
- Implemented both sPCE and sNMC in the log domain. Each candidate parameter vector carries its own state through all selected durations; observed designs are held fixed. The generating parameter is included only in the sPCE denominator. Gaussian normalizers cancel from the likelihood ratios.
- Tested 30 histories through L=128, 10,000, and 100,000: two episodes per method for T=3/4 and one per method for T=5, all from training seed 101/evaluation seed 1001. Also repeated 18 histories in an isolated L=10,000 runtime pilot. This selection is deliberately small and not representative evidence for a method ranking.
- Reproduced archived scores with maximum absolute error below 3.2e-10 nats and terminal states below 1.7e-13, on the selected histories.
- Four independent tests pass: direct likelihood-ratio formulas, carried-state likelihood against manual Gaussian calculations, numerical underflow handling, and chunk-size invariance.

## Runtime on LabPC

Hardware: NVIDIA RTX 4090, 24 GB. Environment: `/home/grads/g/g.lin/miniconda3/envs/dad_mocu_kuramoto/bin/python`, PyTorch 2.5.1+cu121, PyCUDA, NumPy, SciPy, PyYAML, and nvcc on PATH. About 1 GB total GPU memory usage was observed while scoring 10,000-candidate chunks; that includes CUDA/Python context overhead and is not an isolated scorer peak measurement.

The complete IEEE 14-bus archive contains 64,512 histories, 21,504 per horizon: four trained methods x three training seeds x 1536 episodes, plus two untrained baselines x 1536 episodes, for each T. Baselines already appear only once in this archive; no additional duplicate removal was needed.

| T | L=10,000 contrast scoring, seconds/history | L=10,000 including old-score and true-state checks | Full archive at this T, L=10,000 | Projected full archive at this T, L=100,000 |
|---|---:|---:|---:|---:|
| 3 | 0.445 | 1.187 | 7.09 h | 31.00 h |
| 4 | 0.593 | 1.582 | 9.45 h | 41.35 h |
| 5 | 0.745 | 1.987 | 11.87 h | 51.91 h |
| Total | — | — | **28.41 h** | **124.27 h (~5.2 days)** |

The 10,000 estimates extrapolate measured per-history times from six histories per T. The 100,000 estimates add nine additional 10,000-candidate chunks per history to the measured end-to-end cost. Separate 100,000 diagnostic timings were consistent with roughly 4.8–5.4 s for T=3, 6.4–7.2 s for T=4, and about 8 s for most T=5 histories before the extra validation costs. The initial T=3 history and final T=5 Random history briefly overlapped between two diagnostic processes; those inflated timings were excluded from the isolated 10,000 runtime measurement. Estimates are preliminary, not guaranteed wall times; archive read/setup/output overhead, GPU contention, thermal behavior, and checks on later episode indices can change them. The current scorer regenerates old contrasts sequentially through an episode index, which also adds work for later episodes.

## Important numerical finding

L=10,000 is not automatically sufficient. For the T=5 RL-sBOED diagnostic episode 0:

- Archived L=128 sPCE is near the 4.8598-nat ceiling.
- At L=10,000, sPCE=9.2104 nats, nearly the 9.21044-nat ceiling; sNMC=20.2581 nats.
- At L=100,000, sPCE=11.0248 and sNMC=11.9761 nats, a gap of approximately 0.9513 nats.

This is one episode, not a policy mean or ranking. It shows why convergence and upper/lower gaps need checking instead of merely choosing a larger ceiling. sPCE and sNMC bound EIG in expectation; a finite empirical pair is not an EIG confidence interval. Individual estimates need not change monotonically as L grows.

## Implementation requirements established

1. Separate evaluation contrast counts from training/validation, Myopic posterior planning, and Step-DAD refinement. Using the existing `--contrasts` option changes multiple operations and is rejected by the checkpoint-reuse guard. The dedicated rescorer avoids those changes.
2. Use saved `true_MK`, `duration_sequence_s`, `observations_rocof_hz_s`, `true_terminal_state`, evaluation seed, episode index, and resolved physical configuration. No new policy actions or simulated observations are generated.
3. Use the exact archived simulator. The conference-only release moved CUDA context helpers and removed unrelated simulator code, so raw simulator hashes differ from the run metadata. The scorer therefore requires an explicit `--simulator-root` pointing to the original, preserved source snapshot and checks the imported numerical dependencies against saved hashes. That snapshot is available in the original project on LabPC but deliberately omitted from the public conference release. The public tool is not self-contained without that original snapshot or a separately verified equivalent implementation.
4. Use independent prior contrastive samples, paired by rescore seed/evaluation seed/episode across methods and training seeds. Accumulate likelihood sums with logsumexp across chunks. Nested levels share contrastive prefixes.
5. Preserve history identity, source checksums, training seed, evaluation seed, and per-episode results. Baselines with the same identity must have identical data before deduplication. The current CLI checks that condition.
6. Before a multi-day production run, add robust restart/resume with validation of completed records and split work into non-overlapping runs/shards. The pilot writes records incrementally but the CLI currently requires a fresh output directory; it does not yet resume automatically.
7. Consider batching multiple histories, caching per-history true likelihoods/validation results, and persistent CUDA buffers to reduce small-kernel and transfer overhead. Remeasure after optimization. Parallel independent shards on multiple available GPUs could reduce elapsed time without changing the estimator; no extra jobs have been launched.
8. Report paired method differences and uncertainty using the revised scores, with training-seed variability distinguished from conditional episode variability. Do not copy the pilot numbers into the paper's result tables.
9. Check convergence beyond L=10,000 on a larger diagnostic subset before deciding production L. Increasing L to 100,000 improves the demonstrated episode but does not certify convergence. Importance-sampling alternatives would need separate derivation/validation; an unweighted posterior denominator is not a valid substitute for prior marginal likelihood.

## Reproduce a pilot

From the conference repository on LabPC:

```bash
export PATH=/home/grads/g/g.lin/miniconda3/envs/dad_mocu_kuramoto/bin:$PATH
python tools/rescore_saved_histories.py \
  --simulator-root /home/grads/g/g.lin/Documents/objective_power_grid/TPEC_conference/results/experiments/09212026/09212026_ieee14_eig_conference_step4/T5/train101/source_snapshot \
  --runs TPEC_conference/results/experiments/09212026/09212026_ieee14_eig_conference_step4/T5/train101 \
  --output experiments/larger_L_new_pilot \
  --levels 128,10000,100000 --chunk-size 10000 --pilot-per-method 1
```

Setting `--pilot-per-method 0` scores all histories in the specified runs and may take substantial time. No full-archive run has been started.

Pilot data are stored beside this report under `pilot_T5/`, `pilot_T3_T4/`, and `runtime_10000/`. Original raw records, checkpoints, and training code remain untouched.

Estimator definitions: Foster et al. (2021), equations (9)–(13): https://proceedings.mlr.press/v139/foster21a/foster21a.pdf .
