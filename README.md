# DODRL — Data-driven On-Demand Reinforcement Learning for CPU Frequency Governor Control

Dataset, offline learning notebooks, online training, and evaluation for a
config-only DDQN system that tunes the Linux `ondemand` CPU governor on a single
server core of an included **M/G/1-FCFS simulator** (Java client–server
workload) with near-deterministic service times.

## Pipeline

```
available data → integrity checks / cleaning (nb1) → offline expert / student / DDQN (nb2)
              → online training on physical hardware → fixed-duration evaluation
```

1. **Data** — bundled x86 measurements: `MG1/Dataset/x86_mg1.csv`.
2. **Cleaning** — `notebooks/01_MG1_Data_Integrity.ipynb` verifies labels and
   measurement consistency, then exports a cleaned dataset.
3. **Offline learning** — `notebooks/02_CPU_Governor_X86_ConfigOnly.ipynb` fits a
   PCHIP expert and an MLP configuration student, then trains a
   configuration-only DDQN on the student's measured-data surface. The online
   warm-start bundle is `offline/warmstart_bundle.npz`.
4. **Online training** — one warm-started DDQN per arrival rate λ (0.8 … 7.5),
   trained in an RL loop on the live `MG1/` simulator with energy/latency
   weights 0.55 / 0.45. Ten saved models: `models/`.
5. **Evaluation** — saved models run on the live simulator in fixed-duration
   windows; RAPL energy, service time, and throughput are recorded.

## Repository layout

```
agent/        config-only DDQN core (config_only_ddqn.py, cpu_control.py, …)
training/     session runner, warm-start launcher, weight chooser
offline/      offline deployment weights, metadata, warm-start bundle
notebooks/    data integrity / cleaning and offline expert / student / DDQN
MG1/          M/G/1-FCFS simulator (Java client–server), benchmark scripts, dataset
models/       10 trained per-λ models (λ = 0.8 … 7.5) + summaries + per-λ plots
results/      training-session report, per-λ metrics, cross-λ RL plots
evaluation/   fixed-duration evaluator and plot script
```

## Documentation map

| Topic | File |
|---|---|
| M/G/1-FCFS simulator: architecture, simulation flow, hardware preparation, build / run | `MG1/README.md` |
| Static governor sweep scripts | `MG1/x86_benchmark/README.md` |
| Training methodology and historical training results | `results/BENCHMARK_w055-045.md` |
| Per-λ training-session metrics | `results/per_lambda_results.csv` |
| Fixed-duration evaluation protocol and tables | `evaluation/evaluation_mg1/results.md` |

## Key concepts

- **Simulator.** `MG1/` is a Java client–server M/G/1-FCFS queue running on real
  hardware: the client emits exponential arrivals at rate λ; the server
  processes jobs FCFS on one isolated core. Details: `MG1/README.md`.
- **Config-only.** The action space is 50 governor configurations (ten
  `sampling_rate_us` × five `up_threshold_pct`). Frequency scales freely under
  `ondemand`; the agent never sets it.
- **Warm start.** Online training starts from the offline DDQN bundle produced
  by notebook 02 (`offline/warmstart_bundle.npz`).
- **Reward.** Weighted energy/latency, `0.55·energy + 0.45·latency`, clipped.
  Full definition: `results/BENCHMARK_w055-045.md` §1.4.

## Reproduce

```bash
python3 -m pip install -r requirements.txt

# 1. Data cleaning, then offline learning (produces the warm-start bundle).
jupyter notebook notebooks/

# 2. Compile the MG1 simulator (server + client).
javac -cp "MG1/Server/Server/lib/*:." MG1/Server/Server/src/*.java \
    MG1/client/src/*.java -d MG1/build/

# 3. Loopback alias used by the client and server.
sudo ip addr add 10.0.0.2/24 dev lo

# 4. Train per-λ models on physical hardware. The exact historical command
#    (all 10 λ, all flags) is in results/BENCHMARK_w055-045.md §10;
#    batch launcher: training/launch_approachC_warmstart.sh.

# 5. Regenerate the evaluation plots from the supplied measurements.
python3 evaluation/evaluation_mg1/plot_results.py
```

Online runs require Linux CPUFreq control, an isolated worker core, readable
RAPL counters, and the `ondemand` governor. Follow `MG1/README.md` for hardware
preparation. New fixed-duration measurements can be collected with
`evaluation/evaluation_mg1/eval_mg1.py`.
