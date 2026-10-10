#!/usr/bin/env python3
"""Convert the offline notebook artifact (offline/configuration_only_ddqn_weights.npz)
into the SequentialDDQN warm_start bundle format expected by config_only_ddqn.py.

Mapping:
  artifact trunk_W0 (2,48)  -> trunk_W_0              (copied into rows[:2] of 7-dim trunk; rest zero)
  artifact trunk_b0         -> trunk_b_0
  artifact head_W0 (48,48)  -> config_W_0
  artifact head_b0 (48,)    -> config_b_0
  artifact head_W1 (48,50)  -> config_W_1
  artifact head_b1 (50,)    -> config_b_1
  artifact state_mean (1,2) -> state_mu (flatten)
  artifact state_std (1,2)  -> state_sd (flatten)
  n_configs = 50
"""
import numpy as np

SRC = "offline/configuration_only_ddqn_weights.npz"
DST = "offline/warmstart_bundle.npz"

a = np.load(SRC, allow_pickle=False)

# sanity: action grid must be 50 configs
n_configs = int(len(a["sampling_rates_us"]))
assert n_configs == 50, f"expected 50 configs, got {n_configs}"
assert a["head_W1"].shape == (48, 50), a["head_W1"].shape
assert a["trunk_W0"].shape == (2, 48), a["trunk_W0"].shape

bundle = {
    "n_configs": np.asarray(n_configs),
    "trunk_W_0": a["trunk_W0"].astype(np.float64),
    "trunk_b_0": a["trunk_b0"].astype(np.float64),
    "config_W_0": a["head_W0"].astype(np.float64),
    "config_b_0": a["head_b0"].astype(np.float64),
    "config_W_1": a["head_W1"].astype(np.float64),
    "config_b_1": a["head_b1"].astype(np.float64),
    "state_mu": np.asarray(a["state_mean"], dtype=np.float64).reshape(-1),
    "state_sd": np.asarray(a["state_std"], dtype=np.float64).reshape(-1),
    "intended_feature_order": np.asarray(["rho", "log10(1-rho)"], dtype=object),
}

np.savez(DST, **bundle)
print(f"wrote {DST}")
for k in bundle:
    print(f"  {k}: {bundle[k].shape}")