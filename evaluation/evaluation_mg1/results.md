# Results Section — M/G/1 Evaluation

## 1. How the Evaluation Was Run

### 1.1 Overview

The trained per-λ config-only DDQN models were evaluated on the live M/G/1 queue using a fixed, duration-neutral protocol, against the default Linux `ondemand` governor. Everything runs **sequentially** (no parallel runs) to preserve power-measurement integrity.

### 1.2 Testbed

| Component | Setting |
|-----------|---------|
| Server | 12-core Intel server |
| Server | pinned to core 3 (`-Daffinity.reserved=8`) |
| Client | pinned to core 7 |
| Controller (RAPL + cpufreq) | pinned to core 10 (`sudo taskset -c 10`) |
| Energy measurement | RAPL **core domain** (raw Joules, active + idle) |
| Workload | M/G/1 queue, UDP port 9950, 6 FLOPS/iteration |
| Window duration | Fixed 300 s (5 min) |

### 1.3 Baseline

- **Ondemand (default):** CPU frequency freely scales between hardware min (800 MHz) and max (2.1 GHz) using the kernel's default ondemand governor (`sampling_rate = 10 ms`, `up_threshold = 95%`).

### 1.4 Agent

- **Model:** ConfigOnlyDDQN, 50-action space (10 sampling rates × 5 thresholds), frequency free under ondemand.
- **Inference:** Greedy (ε=0), re-decides every 5 s from the live measured state.
- **Training:** 300 windows per λ, warm-started from offline DDQN, interleaved ondemand reference every 10 windows.

### 1.5 Protocol (per λ)

| Mode | Reps | Duration each | Total |
|------|------|---------------|-------|
| Ondemand (default) | 3× | 5 min | 15 min |
| Agent | 10× | 5 min | 50 min |
| **Total per λ** | **13 runs** | | **65 min** |

- 10 λ × 65 min ≈ **10.8 h** for the full sweep (130 five-minute trials).
- All runs sequential; results averaged over reps.

### 1.6 Energy and Metric Computation

Energy is measured **identically** for both modes — raw RAPL core-domain Joules over the 300 s window:

```python
energy_before = rapl.read()                    # RAPL core-domain counter
... window runs (300 s) ...
energy_after = rapl.read()
domain_energy_j = rapl.joules_between(energy_before, energy_after)   # Joules
```

The only difference between modes is the frequency/sampling config applied:

| Mode | Config |
|------|--------|
| Ondemand | `cpu.prepare_config_only(10_000, 95)` + ondemand governor |
| Agent | `cpu.set_config_only(sr, thr)` every 5 s from the model |

**Derived metrics (same formula for all modes):**

| Metric | Definition | Why |
|--------|-----------|-----|
| Power (W) | domain_energy_j / window_seconds | Normalizes out window length |
| Energy-per-job (J) | domain_energy_j / jobs_completed | Normalizes out throughput (fairest) |
| Whole-window energy (J) | domain_energy_j over 5 min | Total energy consumed |
| FLOPS/Watt | total_flops / domain_energy_j | Useful work per Joule |
| Service time (s) | mean calc_ns per job (telemetry) | Raw latency |
| Response time (s) | mean response_ns per job (telemetry) | End-to-end latency |
| Jobs completed | count of telemetry rows | Throughput |

---

## 2. M/G/1 Results (10/10 λ complete)

### 2.1 Absolute Values

| λ | Mode | Power (W) | E/job (J) | svc (ms) | jobs | Whole E (J) |
|---|------|-----------|-----------|----------|------|-------------|
| 0.8 | ondemand | 1.368 | 1.7545 | 122.2 | 235 | 410.7 |
| 0.8 | agent | 1.222 | 1.5382 | 122.8 | 239 | 366.6 |
| 1.6 | ondemand | 1.549 | 0.9553 | 122.0 | 488 | 464.8 |
| 1.6 | agent | 1.492 | 0.9369 | 128.3 | 478 | 447.7 |
| 2.4 | ondemand | 1.891 | 0.7878 | 121.5 | 720 | 567.5 |
| 2.4 | agent | 1.802 | 0.7519 | 124.4 | 720 | 540.8 |
| 3.2 | ondemand | 2.121 | 0.6537 | 121.2 | 974 | 636.6 |
| 3.2 | agent | 2.120 | 0.6439 | 123.7 | 988 | 636.2 |
| 4.0 | ondemand | 2.334 | 0.6064 | 120.9 | 1156 | 700.5 |
| 4.0 | agent | 2.356 | 0.5903 | 124.6 | 1198 | 706.9 |
| 4.8 | ondemand | 2.638 | 0.5583 | 120.8 | 1419 | 791.9 |
| 4.8 | agent | 2.646 | 0.5472 | 125.2 | 1452 | 793.9 |
| 5.6 | ondemand | 2.893 | 0.5324 | 121.0 | 1631 | 868.3 |
| 5.6 | agent | 2.989 | 0.5346 | 124.0 | 1678 | 897.0 |
| 6.4 | ondemand | 3.245 | 0.5077 | 121.0 | 1919 | 974.0 |
| 6.4 | agent | 3.204 | 0.5031 | 122.5 | 1911 | 961.4 |
| 7.2 | ondemand | 3.524 | 0.4971 | 121.1 | 2129 | 1057.6 |
| 7.2 | agent | 3.510 | 0.4962 | 121.8 | 2122 | 1053.2 |
| 7.5 | ondemand | 3.589 | 0.4769 | 121.0 | 2258 | 1077.2 |
| 7.5 | agent | 3.404 | 0.4573 | 121.2 | 2233 | 1021.3 |

### 2.2 Agent vs Ondemand (Gains: + = agent better)

| λ | E_od (J) | E_ag (J) | E_gain | svc_od (ms) | svc_ag (ms) | svc_gain | jobs_od | jobs_ag | jobs_gain | EJ_od (J) | EJ_ag (J) | EJ_gain |
|---|----------|----------|--------|-------------|-------------|----------|---------|---------|-----------|-----------|-----------|---------|
| 0.8 | 410.7 | 366.6 | **+10.7%** | 122.2 | 122.8 | -0.5% | 235 | 239 | +1.9% | 1.7545 | 1.5382 | **+12.3%** |
| 1.6 | 464.8 | 447.7 | **+3.7%** | 122.0 | 128.3 | -5.1% | 488 | 478 | -2.0% | 0.9553 | 0.9369 | **+1.9%** |
| 2.4 | 567.5 | 540.8 | **+4.7%** | 121.5 | 124.4 | -2.3% | 720 | 720 | -0.1% | 0.7878 | 0.7519 | **+4.6%** |
| 3.2 | 636.6 | 636.2 | +0.1% | 121.2 | 123.7 | -2.1% | 974 | 988 | +1.5% | 0.6537 | 0.6439 | **+1.5%** |
| 4.0 | 700.5 | 706.9 | -0.9% | 120.9 | 124.6 | -3.0% | 1156 | 1198 | +3.7% | 0.6064 | 0.5903 | **+2.7%** |
| 4.8 | 791.9 | 793.9 | -0.3% | 120.8 | 125.2 | -3.7% | 1419 | 1452 | +2.3% | 0.5583 | 0.5472 | **+2.0%** |
| 5.6 | 868.3 | 897.0 | -3.3% | 121.0 | 124.0 | -2.4% | 1631 | 1678 | +2.8% | 0.5324 | 0.5346 | -0.4% |
| 6.4 | 974.0 | 961.4 | **+1.3%** | 121.0 | 122.5 | -1.2% | 1919 | 1911 | -0.4% | 0.5077 | 0.5031 | **+0.9%** |
| 7.2 | 1057.6 | 1053.2 | +0.4% | 121.1 | 121.8 | -0.6% | 2129 | 2122 | -0.3% | 0.4971 | 0.4962 | **+0.2%** |
| 7.5 | 1077.2 | 1021.3 | **+5.2%** | 121.0 | 121.2 | -0.2% | 2258 | 2233 | -1.1% | 0.4769 | 0.4573 | **+4.1%** |
| **Mean** | | | **+2.2%** | | | **-2.1%** | | | **+0.8%** | | | **+3.0%** |

### 2.3 FLOPS/Watt (energy efficiency)

FLOPS/Watt = total_flops / domain_energy_j (MFLOPS/W). Reciprocal of E/job (FLOPS ∝ jobs) — useful work per Joule.

| λ | OD | AG | AG vs OD |
|---|----|----|----------|
| 0.8 | 3.86 | 4.44 | **+15.1%** |
| 1.6 | 7.15 | 7.26 | **+1.5%** |
| 2.4 | 8.64 | 9.06 | **+4.9%** |
| 3.2 | 10.41 | 10.56 | **+1.5%** |
| 4.0 | 11.20 | 11.52 | **+2.9%** |
| 4.8 | 12.16 | 12.43 | **+2.3%** |
| 5.6 | 12.78 | 12.73 | -0.4% |
| 6.4 | 13.40 | 13.51 | **+0.8%** |
| 7.2 | 13.71 | 13.71 | -0.0% |
| 7.5 | 14.26 | 14.86 | **+4.2%** |
| **Mean** | | | **+3.3%** |



## 3. Statistical Notes

- All results are mean over reps (3 for the ondemand baseline, 10 for the agent)
- Standard deviation available in runs.csv
- No outlier removal (all runs valid)
- Sequential execution ensures no thermal/power interference between runs
- Whole-window energy gain ≈ power gain because throughput is held roughly constant (±3%)
