"""NumPy implementation of a factorised sequential DDQN.

The model selects a joint action made of:
  * one of 50 Linux ondemand configurations; and
  * one of N legal target-core frequency bins.

It can warm-start from the contextual-bandit network exported by the companion
notebook.  Online updates, however, always use real transitions
(state, action, reward, next_state, done) and a non-zero discount factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


SAMPLING_RATES_US = np.arange(10_000, 100_001, 10_000, dtype=np.int64)
THRESHOLDS_PCT = np.arange(75, 96, 5, dtype=np.int64)
ACTION_GRID = np.asarray(
    [(sr, threshold) for sr in SAMPLING_RATES_US for threshold in THRESHOLDS_PCT],
    dtype=np.int64,
)
SR_NORM = (ACTION_GRID[:, 0] - SAMPLING_RATES_US.min()) / (
    SAMPLING_RATES_US.max() - SAMPLING_RATES_US.min()
)
THR_NORM = (ACTION_GRID[:, 1] - THRESHOLDS_PCT.min()) / (
    THRESHOLDS_PCT.max() - THRESHOLDS_PCT.min()
)
CONFIG_FEATURES = np.column_stack([SR_NORM, THR_NORM])


class MLP:
    """Small tanh MLP with Adam, kept dependency-free for CPU deployment."""

    def __init__(self, sizes: Iterable[int], seed: int, lr: float):
        self.sizes = list(sizes)
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        self.W = [
            rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
            for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.b = [np.zeros(fan_out, dtype=float) for fan_out in self.sizes[1:]]
        self.reset_optimizer()

    def reset_optimizer(self) -> None:
        self.mW = [np.zeros_like(value) for value in self.W]
        self.vW = [np.zeros_like(value) for value in self.W]
        self.mb = [np.zeros_like(value) for value in self.b]
        self.vb = [np.zeros_like(value) for value in self.b]
        self.t = 0

    def forward(self, inputs: np.ndarray, cache: bool = False) -> np.ndarray:
        activation = np.asarray(inputs, dtype=float)
        activations = [activation]
        for layer_index, (weights, bias) in enumerate(zip(self.W, self.b)):
            linear = activation @ weights + bias
            activation = np.tanh(linear) if layer_index < len(self.W) - 1 else linear
            activations.append(activation)
        if cache:
            self._activations = activations
        return activation

    def backward(self, output_gradient: np.ndarray) -> Tuple[list, list, np.ndarray]:
        activations = self._activations
        weight_gradients = [None] * len(self.W)
        bias_gradients = [None] * len(self.b)
        layer_gradient = np.asarray(output_gradient, dtype=float)
        input_gradient = None
        for layer_index in range(len(self.W) - 1, -1, -1):
            weight_gradients[layer_index] = activations[layer_index].T @ layer_gradient
            bias_gradients[layer_index] = layer_gradient.sum(axis=0)
            input_gradient = layer_gradient @ self.W[layer_index].T
            if layer_index > 0:
                layer_gradient = input_gradient * (1.0 - activations[layer_index] ** 2)
        return weight_gradients, bias_gradients, input_gradient

    def adam_step(
        self,
        weight_gradients: list,
        bias_gradients: list,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.t += 1
        for layer_index in range(len(self.W)):
            self.mW[layer_index] = (
                beta1 * self.mW[layer_index]
                + (1.0 - beta1) * weight_gradients[layer_index]
            )
            self.vW[layer_index] = (
                beta2 * self.vW[layer_index]
                + (1.0 - beta2) * weight_gradients[layer_index] ** 2
            )
            self.mb[layer_index] = (
                beta1 * self.mb[layer_index]
                + (1.0 - beta1) * bias_gradients[layer_index]
            )
            self.vb[layer_index] = (
                beta2 * self.vb[layer_index]
                + (1.0 - beta2) * bias_gradients[layer_index] ** 2
            )
            m_weights = self.mW[layer_index] / (1.0 - beta1**self.t)
            v_weights = self.vW[layer_index] / (1.0 - beta2**self.t)
            m_bias = self.mb[layer_index] / (1.0 - beta1**self.t)
            v_bias = self.vb[layer_index] / (1.0 - beta2**self.t)
            self.W[layer_index] -= self.lr * m_weights / (np.sqrt(v_weights) + epsilon)
            self.b[layer_index] -= self.lr * m_bias / (np.sqrt(v_bias) + epsilon)

    def copy(self) -> "MLP":
        clone = MLP(self.sizes, seed=0, lr=self.lr)
        clone.W = [value.copy() for value in self.W]
        clone.b = [value.copy() for value in self.b]
        clone.reset_optimizer()
        return clone

    def soft_update_from(self, source: "MLP", tau: float) -> None:
        for layer_index in range(len(self.W)):
            self.W[layer_index] = (
                (1.0 - tau) * self.W[layer_index] + tau * source.W[layer_index]
            )
            self.b[layer_index] = (
                (1.0 - tau) * self.b[layer_index] + tau * source.b[layer_index]
            )


class FactorisedQNetwork:
    """Q(s, config, frequency) = Q_config + Q_frequency."""

    def __init__(
        self,
        state_dim: int,
        n_frequency_bins: int,
        hidden_dim: int,
        seed: int,
        learning_rate: float,
    ):
        self.state_dim = int(state_dim)
        self.n_frequency_bins = int(n_frequency_bins)
        self.hidden_dim = int(hidden_dim)
        self.trunk = MLP([state_dim, hidden_dim], seed=seed, lr=learning_rate)
        self.config_head = MLP(
            [hidden_dim, hidden_dim, len(ACTION_GRID)],
            seed=seed + 1,
            lr=learning_rate,
        )
        self.frequency_head = MLP(
            [hidden_dim + 2, hidden_dim, n_frequency_bins],
            seed=seed + 2,
            lr=learning_rate,
        )

    def copy(self) -> "FactorisedQNetwork":
        clone = FactorisedQNetwork.__new__(FactorisedQNetwork)
        clone.state_dim = self.state_dim
        clone.n_frequency_bins = self.n_frequency_bins
        clone.hidden_dim = self.hidden_dim
        clone.trunk = self.trunk.copy()
        clone.config_head = self.config_head.copy()
        clone.frequency_head = self.frequency_head.copy()
        return clone

    def all_joint_q(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        hidden = self.trunk.forward(states)
        config_q = self.config_head.forward(hidden)
        batch_size = len(states)
        hidden_repeated = np.repeat(hidden[:, None, :], len(ACTION_GRID), axis=1)
        config_repeated = np.broadcast_to(
            CONFIG_FEATURES[None, :, :], (batch_size, len(ACTION_GRID), 2)
        )
        frequency_inputs = np.concatenate([hidden_repeated, config_repeated], axis=2)
        frequency_q = self.frequency_head.forward(
            frequency_inputs.reshape(-1, self.hidden_dim + 2)
        ).reshape(batch_size, len(ACTION_GRID), self.n_frequency_bins)
        return config_q[:, :, None] + frequency_q

    def greedy_actions(
        self,
        states: np.ndarray,
        config_mask: Optional[np.ndarray] = None,
        frequency_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Joint greedy argmax, optionally excluding masked configs/frequency bins."""
        joint_q = self.all_joint_q(states)
        if config_mask is not None:
            config_mask = np.asarray(config_mask, dtype=bool)
            if config_mask.shape[0] != len(ACTION_GRID):
                raise ValueError(
                    f"config mask has length {config_mask.shape[0]}, expected {len(ACTION_GRID)}"
                )
            joint_q = joint_q.copy()
            joint_q[:, ~config_mask, :] = -1e9
        if frequency_mask is not None:
            frequency_mask = np.asarray(frequency_mask, dtype=bool)
            if frequency_mask.shape[0] != self.n_frequency_bins:
                raise ValueError(
                    f"frequency mask has length {frequency_mask.shape[0]}, "
                    f"expected {self.n_frequency_bins}"
                )
            joint_q = joint_q.copy()
            joint_q[:, :, ~frequency_mask] = -1e9
        flat_actions = joint_q.reshape(len(joint_q), -1).argmax(axis=1)
        config_ids = flat_actions // self.n_frequency_bins
        frequency_ids = flat_actions % self.n_frequency_bins
        return config_ids.astype(np.int64), frequency_ids.astype(np.int64)

    def selected_joint_q(
        self,
        states: np.ndarray,
        config_ids: np.ndarray,
        frequency_ids: np.ndarray,
        cache: bool = False,
    ) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        config_ids = np.asarray(config_ids, dtype=np.int64)
        frequency_ids = np.asarray(frequency_ids, dtype=np.int64)
        hidden = self.trunk.forward(states, cache=cache)
        config_q = self.config_head.forward(hidden, cache=cache)
        frequency_inputs = np.column_stack([hidden, CONFIG_FEATURES[config_ids]])
        frequency_q = self.frequency_head.forward(frequency_inputs, cache=cache)
        row_ids = np.arange(len(states))
        return config_q[row_ids, config_ids] + frequency_q[row_ids, frequency_ids]

    def train_selected(
        self,
        states: np.ndarray,
        config_ids: np.ndarray,
        frequency_ids: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        predictions = self.selected_joint_q(states, config_ids, frequency_ids, cache=True)
        batch_size = len(predictions)
        td_error = predictions - np.asarray(targets, dtype=float)
        selected_gradient = 2.0 * td_error / batch_size

        config_output_gradient = np.zeros((batch_size, len(ACTION_GRID)), dtype=float)
        frequency_output_gradient = np.zeros(
            (batch_size, self.n_frequency_bins), dtype=float
        )
        row_ids = np.arange(batch_size)
        config_output_gradient[row_ids, config_ids] = selected_gradient
        frequency_output_gradient[row_ids, frequency_ids] = selected_gradient

        config_dw, config_db, hidden_from_config = self.config_head.backward(
            config_output_gradient
        )
        frequency_dw, frequency_db, frequency_input_gradient = (
            self.frequency_head.backward(frequency_output_gradient)
        )
        hidden_gradient = hidden_from_config + frequency_input_gradient[:, : self.hidden_dim]
        trunk_dw, trunk_db, _ = self.trunk.backward(hidden_gradient)

        self.config_head.adam_step(config_dw, config_db)
        self.frequency_head.adam_step(frequency_dw, frequency_db)
        self.trunk.adam_step(trunk_dw, trunk_db)
        return float(np.mean(td_error**2))

    def soft_update_from(self, source: "FactorisedQNetwork", tau: float) -> None:
        self.trunk.soft_update_from(source.trunk, tau)
        self.config_head.soft_update_from(source.config_head, tau)
        self.frequency_head.soft_update_from(source.frequency_head, tau)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.config_ids = np.zeros(capacity, dtype=np.int64)
        self.frequency_ids = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.dones = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        config_id: int,
        frequency_id: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.states[index] = state
        self.config_ids[index] = config_id
        self.frequency_ids[index] = frequency_id
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple:
        indices = rng.integers(0, self.size, int(batch_size))
        return (
            self.states[indices],
            self.config_ids[indices],
            self.frequency_ids[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


@dataclass
class DDQNConfig:
    state_dim: int = 7
    n_frequency_bins: int = 32
    hidden_dim: int = 48
    gamma: float = 0.99
    learning_rate: float = 3e-4
    target_tau: float = 0.01
    replay_capacity: int = 50_000
    batch_size: int = 64
    min_replay_size: int = 64
    seed: int = 42
    fmin_khz: int = 800_000
    fmax_khz: int = 2_100_000


class SequentialDDQN:
    def __init__(self, config: DDQNConfig):
        if not 0.0 < config.gamma < 1.0:
            raise ValueError("Sequential gamma must be strictly between zero and one")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.online = FactorisedQNetwork(
            config.state_dim,
            config.n_frequency_bins,
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
    def frequency_centers_khz(self) -> np.ndarray:
        edges = np.linspace(
            self.config.fmin_khz,
            self.config.fmax_khz,
            self.config.n_frequency_bins + 1,
        )
        return 0.5 * (edges[:-1] + edges[1:])

    def action(
        self,
        state: np.ndarray,
        epsilon: float,
        config_mask: Optional[np.ndarray] = None,
        frequency_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, bool, int, int, float, float]:
        """Choose (config_id, frequency_id) under epsilon-greedy action masking.

        ``config_mask`` and ``frequency_mask`` are optional boolean arrays that
        disable individual configurations / frequency bins in both the greedy
        argmax and random exploration.  Returns (config_id, frequency_id,
        exploratory, greedy_config_id, greedy_frequency_id, chosen_q, max_q).
        """
        if config_mask is not None:
            config_mask = np.asarray(config_mask, dtype=bool)
            if config_mask.shape[0] != len(ACTION_GRID):
                raise ValueError(
                    f"config mask has length {config_mask.shape[0]}, expected {len(ACTION_GRID)}"
                )
            if not config_mask.any():
                raise RuntimeError("Config mask excludes every configuration")
        if frequency_mask is not None:
            frequency_mask = np.asarray(frequency_mask, dtype=bool)
            if frequency_mask.shape[0] != self.config.n_frequency_bins:
                raise ValueError(
                    f"frequency mask has length {frequency_mask.shape[0]}, "
                    f"expected {self.config.n_frequency_bins}"
                )
            if not frequency_mask.any():
                raise RuntimeError("Frequency mask excludes every bin")

        states = np.asarray(state, dtype=float)[None, :]
        greedy_config_ids, greedy_frequency_ids = self.online.greedy_actions(
            states, config_mask=config_mask, frequency_mask=frequency_mask
        )
        greedy_config_id = int(greedy_config_ids[0])
        greedy_frequency_id = int(greedy_frequency_ids[0])
        max_q = float(
            self.online.selected_joint_q(
                states, greedy_config_ids, greedy_frequency_ids
            )[0]
        )

        if self.rng.random() < float(epsilon):
            config_pool = (
                np.flatnonzero(config_mask)
                if config_mask is not None
                else np.arange(len(ACTION_GRID))
            )
            frequency_pool = (
                np.flatnonzero(frequency_mask)
                if frequency_mask is not None
                else np.arange(self.config.n_frequency_bins)
            )
            config_id = int(self.rng.choice(config_pool))
            frequency_id = int(self.rng.choice(frequency_pool))
            exploratory = True
        else:
            config_id = greedy_config_id
            frequency_id = greedy_frequency_id
            exploratory = False
        chosen_q = float(
            self.online.selected_joint_q(
                states, np.asarray([config_id]), np.asarray([frequency_id])
            )[0]
        )
        return (
            config_id,
            frequency_id,
            exploratory,
            greedy_config_id,
            greedy_frequency_id,
            chosen_q,
            max_q,
        )

    def encode_observation(
        self,
        utilization: float,
        sampling_rate_us: int,
        threshold_pct: int,
        frequency_khz: float,
        power_ratio: float,
        progress: float,
    ) -> np.ndarray:
        """Build the seven deployment features in the warm-start coordinate system."""
        rho = float(np.clip(utilization, 1e-3, 1.0 - 1e-3))
        legacy_raw = np.asarray([rho, np.log10(max(1.0 - rho, 1e-3))])
        legacy_encoded = (legacy_raw - self.legacy_state_mu) / self.legacy_state_sd
        sr_norm = (float(sampling_rate_us) - SAMPLING_RATES_US.min()) / (
            SAMPLING_RATES_US.max() - SAMPLING_RATES_US.min()
        )
        threshold_norm = (float(threshold_pct) - THRESHOLDS_PCT.min()) / (
            THRESHOLDS_PCT.max() - THRESHOLDS_PCT.min()
        )
        frequency_norm = (float(frequency_khz) - self.config.fmin_khz) / (
            self.config.fmax_khz - self.config.fmin_khz
        )
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
        if len(state) != self.config.state_dim:
            raise ValueError(
                f"encode_observation emits 7 features, configured state_dim={self.config.state_dim}"
            )
        return state

    def observe(
        self,
        state: np.ndarray,
        config_id: int,
        frequency_id: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(
            state, config_id, frequency_id, reward, next_state, done
        )
        self.environment_steps += 1

    def td_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        next_config_ids, next_frequency_ids = self.online.greedy_actions(next_states)
        target_next_q = self.target.selected_joint_q(
            next_states, next_config_ids, next_frequency_ids
        )
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
            states, config_ids, frequency_ids, rewards, next_states, dones = batch
            targets = self.td_targets(rewards, next_states, dones)
            predictions = self.online.selected_joint_q(
                states, config_ids, frequency_ids
            )
            greedy_configs, greedy_freqs = self.online.greedy_actions(states)
            max_q_batch = self.online.selected_joint_q(
                states, greedy_configs, greedy_freqs
            )
            td_error = predictions - targets
            loss = self.online.train_selected(
                states, config_ids, frequency_ids, targets
            )
            self.target.soft_update_from(self.online, self.config.target_tau)
            self.training_steps += 1

            losses.append(loss)
            td_errors.append(td_error)
            td_errors_abs.append(np.abs(td_error))
            target_qs.append(targets)
            chosen_qs.append(predictions)
            max_qs.append(max_q_batch)
            greedy_match.append(
                float(
                    np.mean(
                        (config_ids == greedy_configs)
                        & (frequency_ids == greedy_freqs)
                    )
                )
            )
            history.append(
                {
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
                }
            )
        return {
            "updates": float(len(losses)),
            "loss": float(np.mean(losses)),
            "td_error": float(np.mean(np.concatenate(td_errors))),
            "td_error_abs": float(np.mean(np.concatenate(td_errors_abs))),
            "target_q_mean": float(np.mean(np.concatenate(target_qs))),
            "chosen_q_mean": float(np.mean(np.concatenate(chosen_qs))),
            "max_q_mean": float(np.mean(np.concatenate(max_qs))),
            "q_gap_mean": float(np.mean(np.concatenate(max_qs)) - float(np.mean(np.concatenate(chosen_qs)))),
            "greedy_rate": float(np.mean(greedy_match)),
            "history": history,
        }

    def warm_start(self, bundle_path: Path) -> None:
        bundle = np.load(bundle_path, allow_pickle=False)
        expected_bins = int(bundle["n_frequency_bins"])
        if expected_bins != self.config.n_frequency_bins:
            raise ValueError(
                f"Warm-start has {expected_bins} bins, expected "
                f"{self.config.n_frequency_bins}"
            )
        if int(bundle["n_configs"]) != len(ACTION_GRID):
            raise ValueError("Warm-start action grid is incompatible")

        old_trunk = bundle["trunk_W_0"]
        if old_trunk.shape[1] != self.config.hidden_dim:
            raise ValueError("Warm-start hidden width is incompatible")
        if old_trunk.shape[0] > self.config.state_dim:
            raise ValueError("Warm-start uses more state features than deployment")

        self.online.trunk.W[0].fill(0.0)
        self.online.trunk.W[0][: old_trunk.shape[0]] = old_trunk
        self.online.trunk.b[0] = bundle["trunk_b_0"].copy()
        for index in range(len(self.online.config_head.W)):
            self.online.config_head.W[index] = bundle[f"config_W_{index}"].copy()
            self.online.config_head.b[index] = bundle[f"config_b_{index}"].copy()
        for index in range(len(self.online.frequency_head.W)):
            self.online.frequency_head.W[index] = bundle[f"frequency_W_{index}"].copy()
            self.online.frequency_head.b[index] = bundle[f"frequency_b_{index}"].copy()
        self.online.trunk.reset_optimizer()
        self.online.config_head.reset_optimizer()
        self.online.frequency_head.reset_optimizer()
        self.target = self.online.copy()
        if "state_mu" in bundle.files and "state_sd" in bundle.files:
            self.legacy_state_mu = np.asarray(bundle["state_mu"], dtype=float).reshape(-1)[:2]
            self.legacy_state_sd = np.asarray(bundle["state_sd"], dtype=float).reshape(-1)[:2]

    def save(self, path: Path) -> None:
        path = Path(path)
        arrays: Dict[str, np.ndarray] = {
            "state_dim": np.asarray(self.config.state_dim),
            "n_frequency_bins": np.asarray(self.config.n_frequency_bins),
            "hidden_dim": np.asarray(self.config.hidden_dim),
            "gamma": np.asarray(self.config.gamma),
            "learning_rate": np.asarray(self.config.learning_rate),
            "target_tau": np.asarray(self.config.target_tau),
            "fmin_khz": np.asarray(self.config.fmin_khz),
            "fmax_khz": np.asarray(self.config.fmax_khz),
            "training_steps": np.asarray(self.training_steps),
            "environment_steps": np.asarray(self.environment_steps),
            "legacy_state_mu": self.legacy_state_mu,
            "legacy_state_sd": self.legacy_state_sd,
        }
        for network_name, network in (("online", self.online), ("target", self.target)):
            for module_name, module in (
                ("trunk", network.trunk),
                ("config", network.config_head),
                ("frequency", network.frequency_head),
            ):
                for index, value in enumerate(module.W):
                    arrays[f"{network_name}_{module_name}_W_{index}"] = value
                for index, value in enumerate(module.b):
                    arrays[f"{network_name}_{module_name}_b_{index}"] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> None:
        """Restore every tensor saved by :meth:`save` into this agent.

        Validates dimensions against the constructed config, then copies
        weights for the online and target networks, resets optimizers, and
        restores the learning-step counters.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        bundle = np.load(path, allow_pickle=False)
        expected = {
            "state_dim": self.config.state_dim,
            "n_frequency_bins": self.config.n_frequency_bins,
            "hidden_dim": self.config.hidden_dim,
            "fmin_khz": self.config.fmin_khz,
            "fmax_khz": self.config.fmax_khz,
        }
        mismatched = {
            key: (int(bundle[key]), value)
            for key, value in expected.items()
            if key in bundle.files and int(bundle[key]) != int(value)
        }
        if mismatched:
            details = ", ".join(f"{k}: file={f} != config={c}" for k, (f, c) in mismatched.items())
            raise ValueError(f"checkpoint incompatible: {details}")

        locations = []
        for network_name, network in (("online", self.online), ("target", self.target)):
            for module_name, module in (
                ("trunk", network.trunk),
                ("config", network.config_head),
                ("frequency", network.frequency_head),
            ):
                for index in range(len(module.W)):
                    module.W[index] = bundle[f"{network_name}_{module_name}_W_{index}"].copy()
                    module.b[index] = bundle[f"{network_name}_{module_name}_b_{index}"].copy()
                    locations.append(f"{network_name}/{module_name}/{index}")
        if not locations:
            raise ValueError(f"checkpoint has no network tensors: {path}")

        self.online.trunk.reset_optimizer()
        self.online.config_head.reset_optimizer()
        self.online.frequency_head.reset_optimizer()
        if "training_steps" in bundle.files:
            self.training_steps = int(bundle["training_steps"])
        if "environment_steps" in bundle.files:
            self.environment_steps = int(bundle["environment_steps"])
        if "legacy_state_mu" in bundle.files and "legacy_state_sd" in bundle.files:
            self.legacy_state_mu = np.asarray(bundle["legacy_state_mu"], dtype=float).reshape(-1)[:2]
            self.legacy_state_sd = np.asarray(bundle["legacy_state_sd"], dtype=float).reshape(-1)[:2]


def ensemble_decision(
    nets,
    state: np.ndarray,
    config_mask=None,
    frequency_mask=None,
):
    """Majority vote over each net's greedy joint action.

    ``nets`` is a list of agents (or :class:`SequentialDDQN` instances).
    Ties are broken by the highest mean joint Q across nets.  Returns
    (config_id, frequency_id, greedy_config_id, greedy_frequency_id,
    chosen_q, max_q, vote_count).
    """
    votes: Dict[Tuple[int, int], int] = {}
    q_sum: Dict[Tuple[int, int], float] = {}
    net_state = np.asarray(state, dtype=float)[None, :]
    for net in nets:
        joint_q = net.online.all_joint_q(net_state)[0].copy()
        if config_mask is not None:
            joint_q[~config_mask, :] = -1e9
        if frequency_mask is not None:
            joint_q[:, ~frequency_mask] = -1e9
        flat = int(np.argmax(joint_q))
        n_freq = joint_q.shape[1]
        key = (flat // n_freq, flat % n_freq)
        votes[key] = votes.get(key, 0) + 1
        q_sum[key] = q_sum.get(key, 0.0) + float(joint_q[key[0], key[1]])
    (config_id, frequency_id), vote_count = max(
        votes.items(), key=lambda kv: (kv[1], q_sum[kv[0]])
    )
    chosen_q = q_sum[(config_id, frequency_id)] / len(nets)
    return config_id, frequency_id, config_id, frequency_id, chosen_q, chosen_q, vote_count


def epsilon_by_step(
    step: int,
    start: float = 0.25,
    end: float = 0.02,
    decay_steps: int = 2_000,
) -> float:
    fraction = min(max(step, 0) / max(decay_steps, 1), 1.0)
    return float(start + fraction * (end - start))
