"""Config-only sequential DDQN — agent controls (sampling_rate, up_threshold) only.

The ondemand governor is free to float the target core's frequency across the
full approved range (min → max).  The agent never touches frequency limits.

Action space: 50 discrete (sr, thr) pairs = 10 sampling rates × 5 thresholds.
No frequency head — trunk → config_head only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sequential_ddqn import (
    ACTION_GRID,
    SAMPLING_RATES_US,
    THRESHOLDS_PCT,
    MLP,
    CONFIG_FEATURES,
    DDQNConfig,
    ReplayBuffer,
    epsilon_by_step,
)


class ConfigOnlyQNetwork:
    """Q(s, config) — trunk → config_head.  No frequency head."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        seed: int,
        learning_rate: float,
    ):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.trunk = MLP([state_dim, hidden_dim], seed=seed, lr=learning_rate)
        self.config_head = MLP(
            [hidden_dim, hidden_dim, len(ACTION_GRID)],
            seed=seed + 1,
            lr=learning_rate,
        )

    def copy(self) -> "ConfigOnlyQNetwork":
        clone = ConfigOnlyQNetwork.__new__(ConfigOnlyQNetwork)
        clone.state_dim = self.state_dim
        clone.hidden_dim = self.hidden_dim
        clone.trunk = self.trunk.copy()
        clone.config_head = self.config_head.copy()
        return clone

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for all 50 configs.  Shape: (batch, 50)."""
        states = np.atleast_2d(np.asarray(states, dtype=float))
        hidden = self.trunk.forward(states)
        return self.config_head.forward(hidden)

    def greedy_actions(self, states: np.ndarray, config_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Greedy argmax over 50 configs.  Returns config_ids (batch,)."""
        q = self.q_values(states)
        if config_mask is not None:
            config_mask = np.asarray(config_mask, dtype=bool)
            q = q.copy()
            q[:, ~config_mask] = -1e9
        return q.argmax(axis=1).astype(np.int64)

    def selected_q(
        self,
        states: np.ndarray,
        config_ids: np.ndarray,
        cache: bool = False,
        config_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        config_ids = np.asarray(config_ids, dtype=np.int64)
        hidden = self.trunk.forward(states, cache=cache)
        config_q = self.config_head.forward(hidden, cache=cache)
        if config_mask is not None:
            config_mask = np.asarray(config_mask, dtype=bool)
            config_q = config_q.copy()
            config_q[:, ~config_mask] = -1e9
        return config_q[np.arange(len(states)), config_ids]

    def train_selected(
        self,
        states: np.ndarray,
        config_ids: np.ndarray,
        targets: np.ndarray,
        huber_delta: float = 1.0,
    ) -> float:
        predictions = self.selected_q(states, config_ids, cache=True)
        batch_size = len(predictions)
        td_error = predictions - np.asarray(targets, dtype=float)

        # Huber-like gradient: cap the update at +/- huber_delta so a single
        # outlier target (e.g. the -1e6 invalid-window sentinel) cannot explode
        # the network (MSE gradient = 2*td_error/N is unbounded).
        clipped = np.clip(td_error, -huber_delta, huber_delta)

        output_gradient = np.zeros((batch_size, len(ACTION_GRID)), dtype=float)
        output_gradient[np.arange(batch_size), config_ids] = (
            clipped / batch_size
        )

        head_dw, head_db, hidden_gradient = self.config_head.backward(output_gradient)
        trunk_dw, trunk_db, _ = self.trunk.backward(hidden_gradient)

        self.config_head.adam_step(head_dw, head_db)
        self.trunk.adam_step(trunk_dw, trunk_db)

        # Huber loss (delta=1): 0.5*t² where |t|<=1, else |t|-0.5
        abs_e = np.abs(td_error)
        loss = np.where(abs_e <= huber_delta,
                        0.5 * td_error ** 2,
                        huber_delta * (abs_e - 0.5 * huber_delta))
        return float(np.mean(loss))

    def soft_update_from(self, source: "ConfigOnlyQNetwork", tau: float) -> None:
        self.trunk.soft_update_from(source.trunk, tau)
        self.config_head.soft_update_from(source.config_head, tau)


class ConfigOnlyDDQN:
    """DDQN that picks only (sampling_rate, up_threshold).  Frequency is free."""

    def __init__(self, config: DDQNConfig):
        if not 0.0 < config.gamma < 1.0:
            raise ValueError("gamma must be strictly between zero and one")
        # Override n_frequency_bins to 1 — the network ignores frequency entirely
        self.config = DDQNConfig(
            **{**config.__dict__, "n_frequency_bins": 1}
        )
        self.rng = np.random.default_rng(config.seed)
        self.online = ConfigOnlyQNetwork(
            config.state_dim,
            config.hidden_dim,
            config.seed,
            config.learning_rate,
        )
        self.target = self.online.copy()
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim)
        self.training_steps = 0
        self.environment_steps = 0
        self.legacy_state_mu = np.asarray([0.5, np.log10(0.5)], dtype=float)
        self.legacy_state_sd = np.asarray([0.25, 0.30], dtype=float)

    @property
    def n_frequency_bins(self) -> int:
        return 1  # Not used — frequency is free

    def frequency_centers_khz(self) -> np.ndarray:
        # Placeholder — not used in config-only mode
        return np.array([1400000.0])

    def action(
        self,
        state: np.ndarray,
        epsilon: float,
        config_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, bool, int, float, float]:
        """Choose config_id under epsilon-greedy.  No frequency_id (frequency is free).

        Returns (config_id, exploratory, greedy_config_id, chosen_q, max_q).
        """
        if config_mask is not None:
            config_mask = np.asarray(config_mask, dtype=bool)
            if config_mask.shape[0] != len(ACTION_GRID):
                raise ValueError(
                    f"config mask has length {config_mask.shape[0]}, "
                    f"expected {len(ACTION_GRID)}"
                )
            if not config_mask.any():
                raise RuntimeError("Config mask excludes every configuration")

        states = np.asarray(state, dtype=float)[None, :]
        greedy_config_id = int(self.online.greedy_actions(states, config_mask=config_mask)[0])
        max_q = float(
            self.online.selected_q(states, np.asarray([greedy_config_id]))[0]
        )

        if self.rng.random() < float(epsilon):
            pool = (
                np.flatnonzero(config_mask)
                if config_mask is not None
                else np.arange(len(ACTION_GRID))
            )
            config_id = int(self.rng.choice(pool))
            exploratory = True
        else:
            config_id = greedy_config_id
            exploratory = False

        chosen_q = float(
            self.online.selected_q(states, np.asarray([config_id]))[0]
        )
        return config_id, exploratory, greedy_config_id, chosen_q, max_q

    def encode_observation(
        self,
        utilization: float,
        sampling_rate_us: int,
        threshold_pct: int,
        frequency_khz: float,
        power_ratio: float,
        progress: float,
    ) -> np.ndarray:
        """Build the 7-feature state vector.  Frequency is included for state
        representation (the governor's current frequency matters for EDP) but
        the agent does NOT control it."""
        rho = float(np.clip(utilization, 1e-3, 1.0 - 1e-3))
        legacy_raw = np.asarray([rho, np.log10(max(1.0 - rho, 1e-3))])
        legacy_encoded = (legacy_raw - self.legacy_state_mu) / self.legacy_state_sd
        sr_norm = (float(sampling_rate_us) - SAMPLING_RATES_US.min()) / (
            SAMPLING_RATES_US.max() - SAMPLING_RATES_US.min()
        )
        threshold_norm = (float(threshold_pct) - THRESHOLDS_PCT.min()) / (
            THRESHOLDS_PCT.max() - THRESHOLDS_PCT.min()
        )
        frequency_norm = (float(frequency_khz) - 800_000.0) / (2_100_000.0 - 800_000.0)
        state = np.asarray(
            [
                legacy_encoded[0],
                legacy_encoded[1],
                np.clip(sr_norm, 0.0, 1.0),
                np.clip(threshold_norm, 0.0, 1.0),
                np.clip(frequency_norm, 0.0, 1.0),
                np.clip(power_ratio, 0.0, 3.0) - 1.0,
                np.clip(progress, 0.0, 1.0),
            ],
            dtype=float,
        )
        return state

    def observe(
        self,
        state: np.ndarray,
        config_id: int,
        frequency_id: int,  # Ignored — kept for replay buffer compatibility
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # Store frequency_id = 0 (dummy) since we don't track frequency
        self.replay.add(state, config_id, 0, reward, next_state, done)
        self.environment_steps += 1

    def td_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        next_config_ids = self.online.greedy_actions(next_states)
        target_next_q = self.target.selected_q(next_states, next_config_ids)
        return rewards + self.config.gamma * (1.0 - dones) * target_next_q

    def update(self, updates: int = 1) -> Dict[str, float]:
        sufficient = self.replay.size >= max(
            self.config.min_replay_size, self.config.batch_size
        )
        if not sufficient:
            return {
                "updates": 0.0,
                "loss": float("nan"),
                "td_error": float("nan"),
                "td_error_abs": float("nan"),
                "target_q_mean": float("nan"),
                "chosen_q_mean": float("nan"),
                "max_q_mean": float("nan"),
                "q_gap_mean": float("nan"),
                "greedy_rate": float("nan"),
                "history": [],
            }
        losses = []
        td_errors = []
        td_errors_abs = []
        target_qs = []
        chosen_qs = []
        max_qs = []
        greedy_match = []
        history = []
        for _ in range(int(updates)):
            batch = self.replay.sample(self.config.batch_size, self.rng)
            states, config_ids, _frequency_ids, rewards, next_states, dones = batch
            targets = self.td_targets(rewards, next_states, dones)
            predictions = self.online.selected_q(states, config_ids)
            greedy_configs = self.online.greedy_actions(states)
            max_q_batch = self.online.selected_q(states, greedy_configs)
            td_error = predictions - targets
            loss = self.online.train_selected(states, config_ids, targets)
            self.target.soft_update_from(self.online, self.config.target_tau)
            self.training_steps += 1

            losses.append(loss)
            td_errors.append(td_error)
            td_errors_abs.append(np.abs(td_error))
            target_qs.append(targets)
            chosen_qs.append(predictions)
            max_qs.append(max_q_batch)
            greedy_match.append(
                float(np.mean(config_ids == greedy_configs))
            )
            history.append({
                "step": self.training_steps,
                "loss": float(np.mean(td_error ** 2)),
                "td_error": float(np.mean(td_error)),
                "td_error_abs": float(np.mean(np.abs(td_error))),
                "target_q": float(np.mean(targets)),
                "chosen_q": float(np.mean(predictions)),
                "max_q": float(np.mean(max_q_batch)),
                "q_gap": float(np.mean(max_q_batch - predictions)),
                "greedy_rate": float(greedy_match[-1]),
                "replay_size": self.replay.size,
            })
        return {
            "updates": float(len(losses)),
            "loss": float(np.mean(losses)),
            "td_error": float(np.mean(np.concatenate(td_errors))),
            "td_error_abs": float(np.mean(np.concatenate(td_errors_abs))),
            "target_q_mean": float(np.mean(np.concatenate(target_qs))),
            "chosen_q_mean": float(np.mean(np.concatenate(chosen_qs))),
            "max_q_mean": float(np.mean(np.concatenate(max_qs))),
            "q_gap_mean": (
                float(np.mean(np.concatenate(max_qs)))
                - float(np.mean(np.concatenate(chosen_qs)))
            ),
            "greedy_rate": float(np.mean(greedy_match)),
            "history": history,
        }

    def warm_start(self, bundle_path: Path) -> None:
        """Warm-start from a SequentialDDQN bundle.  Loads trunk + config_head
        weights only; ignores frequency_head."""
        bundle = np.load(bundle_path, allow_pickle=False)
        expected_config_count = int(bundle["n_configs"]) if "n_configs" in bundle.files else len(ACTION_GRID)
        if expected_config_count != len(ACTION_GRID):
            raise ValueError(
                f"Warm-start action grid is incompatible: "
                f"bundle has {expected_config_count}, expected {len(ACTION_GRID)}"
            )

        old_trunk = bundle["trunk_W_0"]
        if old_trunk.shape[1] != self.config.hidden_dim:
            raise ValueError("Warm-start hidden width is incompatible")
        if old_trunk.shape[0] > self.config.state_dim:
            raise ValueError("Warm-start uses more state features than deployment")

        # Load trunk
        self.online.trunk.W[0].fill(0.0)
        self.online.trunk.W[0][: old_trunk.shape[0]] = old_trunk
        self.online.trunk.b[0] = bundle["trunk_b_0"].copy()

        # Load config_head from bundle (compatible with SequentialDDQN format)
        for index in range(len(self.online.config_head.W)):
            key = f"config_W_{index}"
            if key in bundle.files:
                self.online.config_head.W[index] = bundle[key].copy()
            key_b = f"config_b_{index}"
            if key_b in bundle.files:
                self.online.config_head.b[index] = bundle[key_b].copy()

        # Reset optimizers
        self.online.trunk.reset_optimizer()
        self.online.config_head.reset_optimizer()
        self.target = self.online.copy()

        if "state_mu" in bundle.files and "state_sd" in bundle.files:
            self.legacy_state_mu = np.asarray(bundle["state_mu"], dtype=float).reshape(-1)[:2]
            self.legacy_state_sd = np.asarray(bundle["state_sd"], dtype=float).reshape(-1)[:2]

    def save(self, path: Path) -> None:
        path = Path(path)
        arrays: Dict[str, np.ndarray] = {
            "state_dim": np.asarray(self.config.state_dim),
            "n_frequency_bins": np.asarray(1),  # Config-only
            "hidden_dim": np.asarray(self.config.hidden_dim),
            "gamma": np.asarray(self.config.gamma),
            "learning_rate": np.asarray(self.config.learning_rate),
            "target_tau": np.asarray(self.config.target_tau),
            "n_configs": np.asarray(len(ACTION_GRID)),
            "training_steps": np.asarray(self.training_steps),
            "environment_steps": np.asarray(self.environment_steps),
            "legacy_state_mu": self.legacy_state_mu,
            "legacy_state_sd": self.legacy_state_sd,
        }
        for network_name, network in (("online", self.online), ("target", self.target)):
            for module_name, module in (
                ("trunk", network.trunk),
                ("config", network.config_head),
            ):
                for index, value in enumerate(module.W):
                    arrays[f"{network_name}_{module_name}_W_{index}"] = value
                for index, value in enumerate(module.b):
                    arrays[f"{network_name}_{module_name}_b_{index}"] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> None:
        """Restore from a saved checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        bundle = np.load(path, allow_pickle=False)
        expected = {
            "state_dim": self.config.state_dim,
            "hidden_dim": self.config.hidden_dim,
        }
        mismatched = {
            key: (int(bundle[key]), value)
            for key, value in expected.items()
            if key in bundle.files and int(bundle[key]) != int(value)
        }
        if mismatched:
            details = ", ".join(
                f"{k}: file={f} != config={c}"
                for k, (f, c) in mismatched.items()
            )
            raise ValueError(f"checkpoint incompatible: {details}")

        for network_name, network in (("online", self.online), ("target", self.target)):
            for module_name, module in (
                ("trunk", network.trunk),
                ("config", network.config_head),
            ):
                for index in range(len(module.W)):
                    w_key = f"{network_name}_{module_name}_W_{index}"
                    b_key = f"{network_name}_{module_name}_b_{index}"
                    if w_key in bundle.files:
                        module.W[index] = bundle[w_key].copy()
                    if b_key in bundle.files:
                        module.b[index] = bundle[b_key].copy()

        self.online.trunk.reset_optimizer()
        self.online.config_head.reset_optimizer()
        if "training_steps" in bundle.files:
            self.training_steps = int(bundle["training_steps"])
        if "environment_steps" in bundle.files:
            self.environment_steps = int(bundle["environment_steps"])
        if "legacy_state_mu" in bundle.files and "state_sd" in bundle.files:
            self.legacy_state_mu = np.asarray(bundle["state_mu"], dtype=float).reshape(-1)[:2]
            self.legacy_state_sd = np.asarray(bundle["state_sd"], dtype=float).reshape(-1)[:2]


# ---------------------------------------------------------------------------
# Backward-compatible wrapper: adapts ConfigOnlyDDQN to the same interface
# the runner expects from SequentialDDQN, so the runner needs minimal changes.
# ---------------------------------------------------------------------------

class ConfigOnlyWrapper:
    """Drop-in wrapper that presents ConfigOnlyDDQN as if it had the
    SequentialDDQN interface (action returns 7 values, observe takes
    frequency_id, replay, environment_steps, warm_start, etc.) so the
    runner needs minimal changes."""

    def __init__(self, ddqn: ConfigOnlyDDQN):
        self.ddqn = ddqn
        self.config = ddqn.config
        self.trunk = ddqn.online.trunk          # for warm_start compatibility
        self.config_head = ddqn.online.config_head
        self.frequency_head = None              # not used
        self.rng = ddqn.rng                     # runner reads agent.rng
        self.replay = ddqn.replay               # runner reads agent.replay.size
        self.environment_steps = ddqn.environment_steps  # runner reads agent.environment_steps
        self.training_steps = ddqn.training_steps  # runner reads agent.training_steps
        # Runner compatibility: provides frequency_centers_khz attribute
        self.frequency_centers_khz = np.linspace(800000.0, 2100000.0, 16)
        self.n_frequency_bins = 16  # Runner checks this

    def action(
        self,
        state: np.ndarray,
        epsilon: float,
        config_mask: Optional[np.ndarray] = None,
        frequency_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, bool, int, int, float, float]:
        """Returns (config_id, frequency_id=0, exploratory, greedy_config,
        greedy_frequency_id=0, chosen_q, max_q).  frequency_id is always 0
        since we don't control frequency."""
        config_id, exploratory, greedy_config_id, chosen_q, max_q = (
            self.ddqn.action(state, epsilon, config_mask=config_mask)
        )
        return config_id, 0, exploratory, greedy_config_id, 0, chosen_q, max_q

    def encode_observation(self, *args, **kwargs) -> np.ndarray:
        return self.ddqn.encode_observation(*args, **kwargs)

    def observe(self, *args, **kwargs) -> None:
        self.ddqn.observe(*args, **kwargs)

    def update(self, updates: int = 1) -> Dict[str, float]:
        return self.ddqn.update(updates)

    def td_targets(self, *args, **kwargs) -> np.ndarray:
        return self.ddqn.td_targets(*args, **kwargs)

    def save(self, path: Path) -> None:
        self.ddqn.save(path)

    def load(self, path: Path) -> None:
        self.ddqn.load(path)

    def warm_start(self, bundle_path: Path) -> None:
        self.ddqn.warm_start(bundle_path)
