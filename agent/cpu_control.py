"""Linux CPUFreq, utilization, temperature, and RAPL helpers.

All mutations are explicit, logged, and restored on normal exit.  The module is
Linux-only and requires root for real control; dry-run mode performs no writes.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_int(path: Path) -> int:
    return int(read_text(path))


def parse_cpu_list(value: str) -> Set[int]:
    cpus: Set[int] = set()
    for token in value.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(token))
    return cpus


def online_cpu_count(sysfs_root: Path = Path("/sys/devices/system/cpu")) -> int:
    online_path = sysfs_root / "online"
    if online_path.exists():
        return len(parse_cpu_list(read_text(online_path)))
    return os.cpu_count() or 1


def cpu_stat(cpu: int, proc_stat: Path = Path("/proc/stat")) -> Tuple[int, int]:
    prefix = f"cpu{cpu} "
    for line in proc_stat.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            fields = [int(value) for value in line.split()[1:]]
            total = sum(fields)
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            return total, idle
    raise RuntimeError(f"CPU {cpu} is absent from {proc_stat}")


def utilization_between(before: Tuple[int, int], after: Tuple[int, int]) -> float:
    total_delta = after[0] - before[0]
    idle_delta = after[1] - before[1]
    if total_delta <= 0:
        return 0.0
    return min(max(1.0 - idle_delta / total_delta, 0.0), 1.0)


@dataclass
class RaplReading:
    energy_uj: int
    timestamp: float


class RaplDomain:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = read_text(self.path / "name")
        self.energy_path = self.path / "energy_uj"
        self.max_energy_uj = read_int(self.path / "max_energy_range_uj")

    @classmethod
    def discover(
        cls,
        root: Path = Path("/sys/class/powercap"),
        preference: str = "auto",
    ) -> "RaplDomain":
        candidates = []
        queue = [(root, 0)] if root.exists() else []
        visited = set()
        while queue:
            zone, depth = queue.pop(0)
            try:
                resolved = str(zone.resolve())
            except OSError:
                continue
            if resolved in visited:
                continue
            visited.add(resolved)
            name_path = zone / "name"
            energy_path = zone / "energy_uj"
            max_path = zone / "max_energy_range_uj"
            if name_path.exists() and energy_path.exists() and max_path.exists():
                try:
                    candidates.append((read_text(name_path).lower(), zone))
                except (OSError, ValueError):
                    pass
            if depth < 3:
                try:
                    queue.extend(
                        (child, depth + 1)
                        for child in zone.iterdir()
                        if child.is_dir()
                    )
                except OSError:
                    pass
        if not candidates:
            raise RuntimeError("No readable Intel/AMD RAPL energy domain was found")

        def rank(item: tuple) -> tuple:
            name, path = item
            is_mmio = "mmio" in str(path).lower()
            if preference == "core":
                preferred = 0 if ("core" in name or "pp0" in name) else 1
            elif preference == "package":
                preferred = 0 if "package" in name else 1
            else:
                preferred = 0 if ("core" in name or "pp0" in name) else (1 if "package" in name else 2)
            return preferred, is_mmio, len(str(path))

        _, selected = sorted(candidates, key=rank)[0]
        return cls(selected)

    def read(self) -> RaplReading:
        return RaplReading(read_int(self.energy_path), time.monotonic())

    def joules_between(self, before: RaplReading, after: RaplReading) -> float:
        delta_uj = after.energy_uj - before.energy_uj
        if delta_uj < 0:
            delta_uj += self.max_energy_uj
        return delta_uj / 1_000_000.0


class CpuFrequencyController:
    def __init__(
        self,
        target_cpu: int,
        controller_cpu: int,
        requested_min_khz: int = 800_000,
        requested_max_khz: int = 2_100_000,
        sysfs_root: Path = Path("/sys/devices/system/cpu"),
        dry_run: bool = False,
        allow_shared_policy: bool = False,
    ):
        self.target_cpu = int(target_cpu)
        self.controller_cpu = int(controller_cpu)
        self.requested_min_khz = int(requested_min_khz)
        self.requested_max_khz = int(requested_max_khz)
        self.sysfs_root = Path(sysfs_root)
        self.cpufreq_root = self.sysfs_root / "cpufreq"
        self.dry_run = bool(dry_run)
        self.allow_shared_policy = bool(allow_shared_policy)
        self.policies = sorted(self.cpufreq_root.glob("policy*"), key=self._policy_number)
        if not self.policies:
            raise RuntimeError(f"No CPUFreq policies found under {self.cpufreq_root}")
        self.policy_cpus = {policy: self._related_cpus(policy) for policy in self.policies}
        self.target_policy = self._policy_for_cpu(self.target_cpu)
        self.controller_policy = self._policy_for_cpu(self.controller_cpu)
        self._validate_topology()
        self.snapshot = self._take_snapshot()
        self.tuning_snapshot: Dict[Path, str] = {}
        self.mutation_log: List[str] = []

    @staticmethod
    def _policy_number(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else 10**9

    def _related_cpus(self, policy: Path) -> Set[int]:
        for filename in ("related_cpus", "affected_cpus"):
            path = policy / filename
            if path.exists():
                return parse_cpu_list(read_text(path))
        raise RuntimeError(f"Cannot determine CPUs controlled by {policy}")

    def _policy_for_cpu(self, cpu: int) -> Path:
        matches = [policy for policy, cpus in self.policy_cpus.items() if cpu in cpus]
        if len(matches) != 1:
            raise RuntimeError(f"CPU {cpu} maps to {len(matches)} CPUFreq policies")
        return matches[0]

    def _thread_siblings(self, cpu: int) -> Set[int]:
        path = self.sysfs_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        return parse_cpu_list(read_text(path)) if path.exists() else {cpu}

    def _validate_topology(self) -> None:
        if self.target_cpu == self.controller_cpu:
            raise ValueError("Target and controller CPU must be different")
        target_siblings = self._thread_siblings(self.target_cpu)
        if self.controller_cpu in target_siblings:
            raise ValueError(
                f"CPU {self.controller_cpu} is an SMT sibling of target CPU "
                f"{self.target_cpu}; select another physical core"
            )
        controlled_cpus = self.policy_cpus[self.target_policy]
        if controlled_cpus != {self.target_cpu} and not self.allow_shared_policy:
            raise ValueError(
                f"{self.target_policy.name} controls CPUs {sorted(controlled_cpus)}, so "
                "frequency cannot be changed for only the target CPU. Use a per-core "
                "policy or pass --allow-shared-policy with this limitation documented."
            )
        if self.controller_cpu not in os.sched_getaffinity(0):
            raise RuntimeError(
                f"This process is not allowed to run on controller CPU {self.controller_cpu}"
            )

    def assert_process_pinned(self) -> None:
        affinity = os.sched_getaffinity(0)
        if affinity != {self.controller_cpu}:
            raise RuntimeError(
                f"Controller affinity is {sorted(affinity)}, expected only "
                f"CPU {self.controller_cpu}. Launch with taskset -c {self.controller_cpu}."
            )

    def _take_snapshot(self) -> Dict[Path, Dict[str, str]]:
        fields = ["scaling_governor", "scaling_min_freq", "scaling_max_freq"]
        return {
            policy: {
                field: read_text(policy / field)
                for field in fields + (["scaling_setspeed"] if read_text(policy / "scaling_governor") == "userspace" else [])
                if (policy / field).exists()
            }
            for policy in self.policies
        }

    def _write(self, path: Path, value: object) -> None:
        message = f"{path} <- {value}"
        self.mutation_log.append(message)
        if self.dry_run:
            return
        if os.geteuid() != 0:
            raise PermissionError("CPUFreq control requires root")
        path.write_text(f"{value}\n", encoding="utf-8")

    def _available_governors(self, policy: Path) -> Set[str]:
        path = policy / "scaling_available_governors"
        return set(read_text(path).split()) if path.exists() else set()

    def _available_frequencies(self, policy: Path) -> List[int]:
        path = policy / "scaling_available_frequencies"
        if not path.exists():
            return []
        hardware_min, hardware_max = self._hardware_limits(policy)
        return [
            int(value)
            for value in read_text(path).split()
            if hardware_min <= int(value) <= hardware_max
        ]

    def _hardware_limits(self, policy: Path) -> Tuple[int, int]:
        return read_int(policy / "cpuinfo_min_freq"), read_int(policy / "cpuinfo_max_freq")

    def _safe_limits(self, policy: Path, minimum: int, maximum: int) -> None:
        hardware_min, hardware_max = self._hardware_limits(policy)
        minimum = max(hardware_min, min(int(minimum), hardware_max))
        maximum = max(minimum, min(int(maximum), hardware_max))

        def apply() -> None:
            # Widen the range, set the ceiling, then raise the floor.
            # This ordering avoids transient violations where min > max.
            self._write(policy / "scaling_min_freq", hardware_min)
            self._write(policy / "scaling_max_freq", maximum)
            self._write(policy / "scaling_min_freq", minimum)

        def verify() -> bool:
            if self.dry_run:
                return True
            actual_min = read_int(policy / "scaling_min_freq")
            actual_max = read_int(policy / "scaling_max_freq")
            return actual_min == minimum and actual_max == maximum

        for attempt in range(3):
            apply()
            if verify():
                return
            if attempt < 2:
                time.sleep(0.05)
        actual_min = read_int(policy / "scaling_min_freq")
        actual_max = read_int(policy / "scaling_max_freq")
        raise RuntimeError(
            f"Could not set {policy.name} limits to "
            f"min={minimum} max={maximum} (got min={actual_min} max={actual_max})"
        )

    def _tunable_paths(self, filename: str) -> List[Path]:
        candidates = [
            self.cpufreq_root / "ondemand" / filename,
            self.target_policy / "ondemand" / filename,
        ]
        candidates.extend(policy / "ondemand" / filename for policy in self.policies)
        unique: Dict[str, Path] = {}
        for path in candidates:
            if path.exists():
                key = str(path.resolve())
                unique[key] = path
        return list(unique.values())

    def prepare(self, initial_sampling_rate_us: int = 10_000, initial_threshold_pct: int = 95) -> None:
        for policy in self.policies:
            available = self._available_governors(policy)
            hardware_min, _ = self._hardware_limits(policy)
            minimum = max(hardware_min, self.requested_min_khz)
            if policy == self.target_policy:
                if available and "ondemand" not in available:
                    raise RuntimeError(f"ondemand governor is unavailable for {policy.name}")
                self._safe_limits(policy, minimum, minimum)
                self._write(policy / "scaling_governor", "ondemand")
            else:
                if available and "userspace" not in available:
                    raise RuntimeError(f"userspace governor is unavailable for {policy.name}")
                self._safe_limits(policy, minimum, minimum)
                self._write(policy / "scaling_governor", "userspace")
                self._write(policy / "scaling_setspeed", minimum)

        sampling_paths = self._tunable_paths("sampling_rate")
        threshold_paths = self._tunable_paths("up_threshold")
        if not sampling_paths or not threshold_paths:
            raise RuntimeError(
                "ondemand sampling_rate/up_threshold controls were not found after enabling the governor"
            )
        sampling_min_paths = self._tunable_paths("sampling_rate_min")
        if sampling_min_paths:
            kernel_minimum = max(read_int(path) for path in sampling_min_paths)
            if kernel_minimum > 10_000:
                raise RuntimeError(
                    f"Kernel ondemand sampling_rate_min={kernel_minimum} us, but the "
                    "trained action grid begins at 10000 us"
                )
        for path in sampling_paths + threshold_paths:
            if path not in self.tuning_snapshot:
                self.tuning_snapshot[path] = read_text(path)
        self.set_global_config(initial_sampling_rate_us, initial_threshold_pct)

    def prepare_config_only(self, initial_sampling_rate_us: int = 10_000, initial_threshold_pct: int = 95) -> None:
        """Set up the target CPU for config-only mode: ondemand governor with
        full frequency range (min → max), never touched by the agent after setup.

        The agent only calls set_global_config(sr, thr) to change ondemand
        parameters.  Frequency floats freely between min and max based on the
        governor's own load-proportional logic."""
        for policy in self.policies:
            available = self._available_governors(policy)
            hardware_min, hardware_max = self._hardware_limits(policy)
            minimum = max(hardware_min, self.requested_min_khz)
            maximum = min(hardware_max, self.requested_max_khz)
            if policy == self.target_policy:
                if available and "ondemand" not in available:
                    raise RuntimeError(f"ondemand governor is unavailable for {policy.name}")
                # Full floating range: min → max, governor decides frequency
                self._safe_limits(policy, minimum, maximum)
                self._write(policy / "scaling_governor", "ondemand")
            else:
                if available and "userspace" not in available:
                    raise RuntimeError(f"userspace governor is unavailable for {policy.name}")
                self._safe_limits(policy, minimum, minimum)
                self._write(policy / "scaling_governor", "userspace")
                self._write(policy / "scaling_setspeed", minimum)
        self.set_global_config(initial_sampling_rate_us, initial_threshold_pct)

    def set_global_config(self, sampling_rate_us: int, threshold_pct: int) -> None:
        sampling_rate_us = int(sampling_rate_us)
        threshold_pct = int(threshold_pct)
        if not 10_000 <= sampling_rate_us <= 100_000:
            raise ValueError("sampling_rate_us must be between 10000 and 100000")
        if threshold_pct not in {75, 80, 85, 90, 95}:
            raise ValueError("threshold_pct must be one of 75, 80, 85, 90, 95")
        for path in self._tunable_paths("sampling_rate"):
            self._write(path, sampling_rate_us)
        for path in self._tunable_paths("up_threshold"):
            self._write(path, threshold_pct)
        if not self.dry_run:
            for path in self._tunable_paths("sampling_rate"):
                if read_int(path) != sampling_rate_us:
                    raise RuntimeError(f"Kernel did not apply sampling_rate={sampling_rate_us} at {path}")
            for path in self._tunable_paths("up_threshold"):
                if read_int(path) != threshold_pct:
                    raise RuntimeError(f"Kernel did not apply up_threshold={threshold_pct} at {path}")

    def set_config_only(self, sampling_rate_us: int, threshold_pct: int) -> None:
        """Set ondemand (sr, thr) without touching frequency limits.

        Use this in config-only mode where the governor floats freely.
        """
        self.set_global_config(sampling_rate_us, threshold_pct)

    def map_frequency(self, requested_khz: float) -> int:
        hardware_min, hardware_max = self._hardware_limits(self.target_policy)
        requested = int(round(float(requested_khz)))
        available_path = self.target_policy / "scaling_available_frequencies"
        if available_path.exists():
            available = [int(value) for value in read_text(available_path).split()]
            available = [value for value in available if hardware_min <= value <= hardware_max]
            if available:
                return min(available, key=lambda value: abs(value - requested))
        return max(hardware_min, min(requested, hardware_max))

    def apply_action(
        self,
        sampling_rate_us: int,
        threshold_pct: int,
        requested_frequency_khz: float,
    ) -> int:
        self.set_global_config(sampling_rate_us, threshold_pct)
        mapped_frequency = self.map_frequency(requested_frequency_khz)
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        # The cap is a CEILING: the governor floats freely between the
        # experiment minimum and mapped_frequency.  The agent controls the
        # ceiling; the governor handles dynamic scaling within that range.
        self._safe_limits(self.target_policy, minimum, mapped_frequency)
        if self.dry_run:
            return mapped_frequency
        effective_maximum = read_int(self.target_policy / "scaling_max_freq")
        effective_minimum = read_int(self.target_policy / "scaling_min_freq")
        if effective_maximum != mapped_frequency or effective_minimum != minimum:
            raise RuntimeError(
                f"Kernel did not set cap range "
                f"min={minimum} max={mapped_frequency} KHz "
                f"(got min={effective_minimum} max={effective_maximum})"
            )
        return effective_maximum

    def apply_baseline_ondemand(
        self,
        sampling_rate_us: int,
        threshold_pct: int,
        max_frequency_khz: float,
    ) -> int:
        """Run the target policy as TRUE ondemand, floating min -> max.

        Unlike apply_action() (which pins min=max to a fixed frequency), the
        baseline governor must scale up and down across the whole range exactly
        as the stock kernel governor would: min = requested experiment minimum,
        max = mapped max frequency, with the given sampling rate / threshold.
        """
        self.set_global_config(sampling_rate_us, threshold_pct)
        mapped_max = self.map_frequency(max_frequency_khz)
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        self._safe_limits(self.target_policy, minimum, mapped_max)
        self._write(self.target_policy / "scaling_governor", "ondemand")
        if self.dry_run:
            return mapped_max
        effective_maximum = read_int(self.target_policy / "scaling_max_freq")
        effective_minimum = read_int(self.target_policy / "scaling_min_freq")
        if effective_maximum != mapped_max or effective_minimum != minimum:
            raise RuntimeError(
                f"Kernel did not set floating ondemand range "
                f"min={minimum} max={mapped_max} KHz "
                f"(got min={effective_minimum} max={effective_maximum})"
            )
        return effective_maximum

    def apply_performance_governor(self, max_frequency_khz: float) -> int:
        """Pin the target policy to the performance governor at max frequency.

        Used as the "always at max frequency" baseline: the core never
        down-clocks regardless of load.  Limits are pinned to the mapped
        maximum so the governor has no room to lower the P-state.
        """
        mapped_max = self.map_frequency(max_frequency_khz)
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        self._safe_limits(self.target_policy, minimum, mapped_max)
        self._write(self.target_policy / "scaling_governor", "performance")
        if self.dry_run:
            return mapped_max
        effective_maximum = read_int(self.target_policy / "scaling_max_freq")
        effective_minimum = read_int(self.target_policy / "scaling_min_freq")
        effective_governor = read_text(self.target_policy / "scaling_governor").strip()
        if (
            effective_maximum != mapped_max
            or effective_minimum != minimum
            or effective_governor != "performance"
        ):
            raise RuntimeError(
                f"Kernel did not apply performance governor "
                f"(min={effective_minimum} max={effective_maximum} "
                f"governor={effective_governor!r}, wanted max={mapped_max})"
            )
        return effective_maximum

    def set_target_userspace_fixed(self, frequency_khz: int) -> int:
        """Lock the target policy to userspace at a fixed frequency.

        Used for idle-power calibration so the target core is guaranteed to
        stay at the requested frequency.
        """
        mapped_frequency = self.map_frequency(frequency_khz)
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        self._safe_limits(self.target_policy, minimum, mapped_frequency)
        self._write(self.target_policy / "scaling_governor", "userspace")
        self._write(self.target_policy / "scaling_setspeed", mapped_frequency)
        if self.dry_run:
            return mapped_frequency
        # Verify the requested setspeed was accepted; scaling_cur_freq can be
        # transient, so allow a short settle window before we complain.
        setspeed_path = self.target_policy / "scaling_setspeed"
        effective_setspeed = read_int(setspeed_path) if setspeed_path.exists() else None
        if effective_setspeed is not None and effective_setspeed != mapped_frequency:
            raise RuntimeError(
                f"Kernel did not accept setspeed={mapped_frequency} KHz "
                f"(got {effective_setspeed} KHz)"
            )
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            effective = read_int(self.target_policy / "scaling_cur_freq")
            if effective == mapped_frequency:
                return effective
            time.sleep(0.05)
        # Allow a small hardware rounding tolerance before failing.
        effective = read_int(self.target_policy / "scaling_cur_freq")
        hardware_min, hardware_max = self._hardware_limits(self.target_policy)
        available = self._available_frequencies(self.target_policy)
        if available and effective not in available:
            raise RuntimeError(
                f"Target core did not lock to {mapped_frequency} KHz "
                f"(got {effective} KHz; available={sorted(available)})")
        return effective

    def restore_target_ondemand(self) -> None:
        """Return the target policy to Ondemand at the experiment minimum."""
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        self._safe_limits(self.target_policy, minimum, minimum)
        self._write(self.target_policy / "scaling_governor", "ondemand")

    def saved_ondemand_defaults(self) -> Tuple[int, int]:
        """Return the Ondemand tunables captured before the experiment changed them."""
        sampling_values = {
            int(value)
            for path, value in self.tuning_snapshot.items()
            if path.name == "sampling_rate"
        }
        threshold_values = {
            int(value)
            for path, value in self.tuning_snapshot.items()
            if path.name == "up_threshold"
        }
        if len(sampling_values) != 1 or len(threshold_values) != 1:
            raise RuntimeError(
                "Could not identify one consistent saved Ondemand default: "
                f"sampling_rate={sorted(sampling_values)}, "
                f"up_threshold={sorted(threshold_values)}"
            )
        return sampling_values.pop(), threshold_values.pop()

    def apply_saved_ondemand_defaults(self, requested_frequency_khz: float) -> Tuple[int, int, int]:
        """Restore captured Ondemand tunables while retaining the experiment's cap."""
        sampling_rate_us, threshold_pct = self.saved_ondemand_defaults()
        for path, value in self.tuning_snapshot.items():
            self._write(path, value)
        mapped_frequency = self.map_frequency(requested_frequency_khz)
        minimum, _ = self._hardware_limits(self.target_policy)
        minimum = max(minimum, self.requested_min_khz)
        self._safe_limits(self.target_policy, minimum, mapped_frequency)
        if not self.dry_run:
            effective_maximum = read_int(self.target_policy / "scaling_max_freq")
            if effective_maximum != mapped_frequency:
                raise RuntimeError(
                    f"Kernel applied target max {effective_maximum} KHz, requested {mapped_frequency} KHz"
                )
        return sampling_rate_us, threshold_pct, mapped_frequency

    def current_frequency_khz(self) -> int:
        for path in (
            self.target_policy / "scaling_cur_freq",
            self.sysfs_root / f"cpu{self.target_cpu}" / "cpufreq" / "scaling_cur_freq",
        ):
            if path.exists():
                return read_int(path)
        raise RuntimeError("Current target CPU frequency is unavailable")

    def restore(self) -> None:
        errors = []
        for path, value in self.tuning_snapshot.items():
            try:
                if not path.exists():
                    # Global Ondemand tunables can disappear when governors change.
                    continue
                self._write(path, value)
            except OSError as error:
                errors.append(f"{path}: {error}")
        for policy, values in self.snapshot.items():
            try:
                minimum = int(values.get("scaling_min_freq", self._hardware_limits(policy)[0]))
                maximum = int(values.get("scaling_max_freq", self._hardware_limits(policy)[1]))
                self._safe_limits(policy, minimum, maximum)
                if "scaling_governor" in values:
                    self._write(policy / "scaling_governor", values["scaling_governor"])
            except OSError as error:
                errors.append(f"{policy}: {error}")
        if errors:
            raise RuntimeError("CPUFreq restore had errors: " + "; ".join(errors))

    def topology_summary(self) -> dict:
        target_min, target_max = self._hardware_limits(self.target_policy)
        return {
            "target_cpu": self.target_cpu,
            "controller_cpu": self.controller_cpu,
            "target_policy": self.target_policy.name,
            "target_policy_cpus": sorted(self.policy_cpus[self.target_policy]),
            "controller_policy": self.controller_policy.name,
            "controller_policy_cpus": sorted(self.policy_cpus[self.controller_policy]),
            "target_hardware_min_khz": target_min,
            "target_hardware_max_khz": target_max,
            "policy_count": len(self.policies),
        }


def read_max_temperature_c(
    thermal_root: Path = Path("/sys/class/thermal"),
    hwmon_root: Path = Path("/sys/class/hwmon"),
) -> Optional[float]:
    values: List[float] = []
    paths = list(thermal_root.glob("thermal_zone*/temp"))
    paths.extend(hwmon_root.glob("hwmon*/temp*_input"))
    for path in paths:
        try:
            value = float(read_text(path))
            value = value / 1000.0 if value > 1_000 else value
            if 0.0 < value < 150.0:
                values.append(value)
        except (OSError, ValueError):
            pass
    return max(values) if values else None
