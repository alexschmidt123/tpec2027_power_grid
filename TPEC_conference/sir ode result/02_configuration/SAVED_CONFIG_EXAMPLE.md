# Example saved configuration (T=3, train=101, eval=1001)

This example is explanatory; use the per-evaluation configurations for each cell.

```json
{
  "sir_ode": {
    "population": 500.0,
    "i0": 2.0,
    "t_end": 100.0,
    "n_grid": 10000,
    "beta_mean": 0.5,
    "gamma_mean": 0.1,
    "log_std": 0.5,
    "min_mean_infected": 1.0,
    "n_actions": 100,
    "likelihood_sigma": 1.0
  },
  "prior": {
    "mc_samples": 1024,
    "mc_support_seed": 1
  },
  "training": {
    "policy_hidden": 256,
    "device": "auto",
    "eig_epochs": 200,
    "eig_steps_per_epoch": 512,
    "batch_size": 256,
    "learning_rate": 0.0003,
    "entropy_coef": 0.05,
    "eig_validation_systems": 256,
    "eig_bc_trajectories": 128,
    "eig_bc_lookahead": "two_step",
    "eig_bc_temperature": 0.5,
    "eig_bc_fantasies": 16,
    "eig_rl_use_ppo": true,
    "eig_prefer_unique_sequence_floor": true,
    "eig_min_unique_sequence_fraction": 0.08,
    "eig_unique_floor_slack": 0.03
  },
  "data_generation": {
    "theta_sample_size_train": 10000,
    "theta_sample_size_test": 3000,
    "train_seed": 101,
    "test_seed": 202,
    "reused": true,
    "force": false,
    "smoke": false,
    "bank_shape_train": null,
    "bank_shape_test": null,
    "elapsed_seconds": 0.0,
    "train_theta_count": 10000,
    "test_theta_count": 3000
  },
  "observation": {
    "noise_sigma": 1.0,
    "sampling": "measurement_time",
    "N_obs": 1
  },
  "evaluation": {
    "eig_test_systems": 512
  }
}
```
