# Detailed SIR-ODE experiment configuration

## Coverage

- Six methods: DAD, RL-sBOED, Step-DAD, Myopic, Fixed, Random. No MoE.
- Horizons: 3, 4, 5. Training seeds: 101, 202, 303. Evaluation seeds: 1001–1005.
- 512 test systems per evaluation; 1,024 prior/posterior-support particles, support seed 1.
- Observations: infected-count measurement, one scalar, Gaussian likelihood sigma 1.0.
- Population 500, initial infected count 2; simulation end 100; 10,000 integration-grid points; 100 candidate observation times. Measurement times must increase.
- Lognormal parameter prior configured by beta_mean=0.5, gamma_mean=0.1, log_std=0.5. Original simulator/config governs exact parameterization.
- Shared simulated bank: 10,000 training and 3,000 test parameter realizations, generation seeds 101 and 202. Evaluation selects 512 systems. Bank binaries are not copied into this result package.
- Training settings: policy hidden width 256; configured 200 epochs × 512 steps/epoch; batch size 256; learning rate 0.0003; entropy coefficient 0.05; 256 validation systems. Actual checkpoint selection and histories are exported beside each .pth.
- Behavioral-cloning warm start: 128 trajectories, two-step lookahead, temperature 0.5, 16 fantasies. These are historical SIR adaptations, not the grid pathwise DAD/REDQ implementation.
- DAD checkpoint metadata identifies REINFORCE; RL metadata records its optimizer separately. Consult ../03_models/checkpoint_inventory.json for exact per-model settings.
- Saved selection configuration includes eig_prefer_unique_sequence_floor=true, fraction 0.08, slack 0.03. Preserve and disclose these historical settings; this package does not relabel them as a newer training protocol.
- Fixed is a calibrated frozen sequence (greedy_prior_eig_frozen), calibration seed 104729, 16 fantasies, independent of evaluation seed. Step-DAD starts from its matching DAD checkpoint; it has no separate offline checkpoint. Myopic and Random have no trained checkpoint.
- Random uses 32 design replicates per system; other methods use one. Verification averages replicates within system before comparing summary means.

## Authoritative records and caveats

`detailed_configurations.json` contains all 45 original resolved evaluation configurations; each run retains the originals byte-for-byte. Original paths and legacy labels are preserved, including obsolete source_config paths and generic observation labels. The evaluated observation is confirmed by vector_eig_results.json (`sir_infected_count`). A reused-data `smoke` field is not used to infer training size; consult checkpoint histories and actual evaluation records.

`../03_models/T*/train_seed_*/*.metadata.json` exports checkpoint metadata, training history, selection and elapsed time without weights. `.pth` files retain weights and all original contents. The package contains available historical artifacts, not a fabricated historical source snapshot. Exact historical source revision/hardware is not recorded consistently; consult ../05_provenance/source_map.json and original logs. Do not pool runtime as hardware-independent evidence.

## Files

- [All original resolved configurations](detailed_configurations.json)
- [Example saved configuration](SAVED_CONFIG_EXAMPLE.md)
- [Model index](../03_models/README.md)
- [Run index](../04_runs/README.md)
