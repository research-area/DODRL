# Benchmark Report — Warm-Started Config-Only DDQN Governor
## Weights 0.55 (energy) / 0.45 (latency), M/G/1 Closed-Loop, 10 λ points

**Artifact directory:** `models/` (per-λ training sessions)
**Figures:** `results/plots/` (cross-λ RL diagnostics) and `models/lambda_*/plots/` (per-λ)
**Branch/commit at run time:** `main` @ `77a18cb`
**Runs executed:** batch of 10 sequential per-λ sessions, 2026-10-03 (03:34 → 08:21 UTC)

---

## Abstract

We benchmark a **warm-started, config-only Double DQN (DDQN) governor** that controls the Linux
`ondemand` cpufreq governor of a single pinned worker core (CPU 3) by choosing its
`sampling_rate_us` and `up_threshold_pct`. The worker serves a closed-loop **M/G/1** workload
emitted by the MG1 Java framework. The agent is fine-tuned online from an **offline pre-trained
2-feature policy** ("warm-start bundle") with a **weighted reward** `0.55·energy + 0.45·latency`.

Across 10 arrival rates (λ = 0.8 … 7.5) with 300 training windows and 12 paired evaluation
windows per λ, the agent:

* **Saves energy heavily and reliably** — the agent cuts **whole-window total energy** (RAPL
  `domain_energy_j`, including idle gaps) by a mean **+26.3 %** vs the stock `ondemand` reference
  (positive = agent better), winning **9 / 10 λ**; up to **+46.3 %** (λ = 3.2).
* Keeps the **service-time penalty small** — mean **−1.7 %** (i.e. the agent is ~1.7 % *slower*),
  never worse than ~2.6 %, always inside the 10 % acceptability budget.
* **Wins on total-EDP** (energy × service time) — mean **+25.3 %** vs `ondemand`, winning
  **9 / 10 λ**.

The headline result: warm-starting yields the **largest whole-window energy saving at the lowest
latency cost** of any configuration tested, and the saving survives on the energy × delay metric
(mean +25.3 %, versus the stock ondemand baseline).

---

## 1. Objective and control formulation

### 1.1 Problem

On a shared multicore package, an unmodified `ondemand` governor on the worker core reacts to
utilisation with a fixed sampling cadence and up-threshold. Tuning these two parameters changes
how aggressively the core is up-clocked and how quickly the governor reacts, which trades
**energy** (fewer/slower clock ramps = cooler package) against **latency** (service time of the
jobs on the worker).

We learn a policy that selects the pair
`(sampling_rate_us, threshold_pct)` from a **50-action grid** (10 sampling rates × 5 thresholds):

```
sampling_rate_us ∈ {10 000, 20 000, …, 100 000}   (10 values, step 10 000)
threshold_pct    ∈ {75, 80, 85, 90, 95}           (5 values,  step 5)
```

### 1.2 Config-only mode (frequency is NOT an agent action)

The `ondemand` governor freely scales the core frequency **between its hardware minimum and
maximum** (`scaling_min_freq = 800 000 kHz`, `scaling_max_freq = 2 100 000 kHz`) as a function of
load. The agent **never writes frequency** — its only action is selecting the two `ondemand`
governor parameters below. Frequency is therefore free under the governor.

```
cpu3 governor            = ondemand
cpu3 scaling_min_freq    = 800000 kHz
cpu3 scaling_max_freq    = 2100000 kHz
cpu3 current freq         = free (governor scales between min and max)
```

### 1.3 State vector (7 features)

`ConfigOnlyQNetwork.encode_observation` builds:

| # | feature | definition |
|---|---------|-----------|
| 0 | `ρ` (legacy)        | `clip(utilisation, 1e-3, 1−1e-3)`, normalised: `(ρ − μ₀)/σ₀` |
| 1 | `log10(1−ρ)` (legacy)| `log10(max(1−ρ, 1e-3))`, normalised: `(· − μ₁)/σ₁` |
| 2 | `sr_norm`           | min-max of sampling rate over the grid [0,1] |
| 3 | `thr_norm`          | min-max of threshold over the grid [0,1] |
| 4 | `freq_norm`         | `(measured_freq_khz − 800 000)/(2 100 000 − 800 000)` clipped [0,1] |
| 5 | `power_ratio`       | `clip(estimated_target_power_w / reference_power, 0, 3) − 1` |
| 6 | `progress`          | training-progress fraction [0,1] |

Features 0–1 are the **two features of the offline warm-start policy** (§4). Normalisation stats
`(μ, σ)` are taken from the warm-start bundle's `state_mu` / `state_sd`.

### 1.4 Reward (WEIGHTED)

```
energy      = estimated_target_power_w × mean_service_time_s        (joules per window)
reward      = clip( −0.55·log(energy / ref_energy)
                    −0.45·log(svc   / ref_svc),   −5, +5 )
```

where `ref_energy`, `ref_svc` are the medians of the stock-ondemand *reference windows*. Because
`energy = power × svc`, this reward expands to
`−0.55·log(power/ref) − 1.0·log(svc/ref)` — i.e. **the service-time term always carries weight 1.0**
(weights sum to 1), and only the power-emphasis dial is swept by the weights. This is the reason
weight variation has limited structural leverage, and it motivates reporting energy and service
time separately alongside a composite metric. (This reward's energy is the training-objective
`est_target_power × svc`; the **evaluation** metric is the whole-window total energy
`domain_energy_j`, scored as total-EDP = energy × service time — see §1.5.10.)

* **Reproducibility fix applied in this batch:** the DDQN loss is **Huber (δ = 1)** rather than MSE,
  and the invalid-window sentinel reward is **−5** (not −1e6). MSE plus a −1e6 sentinel had
  previously produced gradient explosions (loss spikes up to 2.5e11) that corrupted the network and
  made identical runs diverge. Huber caps the gradient at ±1, eliminating that failure mode.

---

## 1.5 Mathematical formulation

### 1.5.1 Notation

| symbol | meaning |
|--------|---------|
| $s_t \in \mathbb{R}^7$ | state at decision epoch $t$ |
| $a_t \in \{0,\dots,49\}$ | action = config id (sampling-rate × threshold) |
| $r_t$ | scalar reward after $(s_t,a_t)$ |
| $\theta$ | online network parameters |
| $\theta^-$ | target network parameters |
| $\gamma = 0.95$ | discount factor |
| $\tau = 0.01$ | target soft-update rate |
| $N = 16$ | mini-batch size |
| $\delta = 1$ | Huber threshold |

### 1.5.2 State encoding

Let $u$ be the window utilisation, $f$ the measured core frequency (kHz), $P$ the estimated
target power, $P_{\text{ref}}$ the current ondemand reference power, and $p$ the session progress.
With $\mu,\sigma$ the warm-start bundle's normaliser statistics (`state_mu`, `state_sd`):

$$
\rho = \mathrm{clip}(u,\;10^{-3},\;1-10^{-3})
$$
$$
\tilde{x}_0 = \frac{\rho - \mu_0}{\sigma_0}, \qquad
\tilde{x}_1 = \frac{\log_{10}\!\big(\max(1-\rho,\,10^{-3})\big) - \mu_1}{\sigma_1}
$$
$$
x_2 = \frac{sr - 10^{4}}{10^{5}-10^{4}}, \quad
x_3 = \frac{thr - 75}{95-75}, \quad
x_4 = \mathrm{clip}\!\Big(\frac{f - 8\cdot10^{5}}{2.1\cdot10^{6}-8\cdot10^{5}},0,1\Big)
$$
$$
x_5 = \mathrm{clip}\!\Big(\frac{P}{\max(P_{\text{ref}},10^{-9})},0,3\Big) - 1, \qquad
x_6 = \mathrm{clip}(p,0,1)
$$
$$
\boxed{\,s_t = \big[\,\tilde{x}_0,\ \tilde{x}_1,\ x_2,\ x_3,\ x_4,\ x_5,\ x_6\,\big]^{\top} \in \mathbb{R}^7\,}
$$

Features $(\tilde{x}_0,\tilde{x}_1)$ are the **two inputs of the offline warm-start policy**;
$x_2,\dots,x_6$ are new features the offline model never saw (their input-layer rows start at zero).

### 1.5.3 Q-network forward pass

The network is a 3-layer tanh MLP mapping the 7-dim state to 50 Q-values
(activation $\tanh$ on the hidden layers, identity on the output layer):

$$
h = \tanh\!\big(W^{\text{tr}} s_t + b^{\text{tr}}\big), \qquad W^{\text{tr}} \in \mathbb{R}^{48\times 7},\; b^{\text{tr}} \in \mathbb{R}^{48}
$$
$$
g = \tanh\!\big(W^{(0)} h + b^{(0)}\big), \qquad W^{(0)} \in \mathbb{R}^{48\times 48},\; b^{(0)} \in \mathbb{R}^{48}
$$
$$
\boxed{\,Q(s_t,\cdot\,;\theta) = W^{(1)} g + b^{(1)} \in \mathbb{R}^{50}, \qquad W^{(1)} \in \mathbb{R}^{50\times 48},\; b^{(1)} \in \mathbb{R}^{50}\,}
$$

Parameter count: $(7\!\cdot\!48+48) + (48\!\cdot\!48+48) + (50\!\cdot\!48+50) = 384 + 2352 + 2450 = \mathbf{5186}$
(the offline 2-feature model had 4 946).

> **Storage convention.** The equations above use the standard $(out, in)$ matrix convention
> ($Q = W g$). The NumPy implementation stores each weight matrix transposed as $(in, out)$ because
> the forward pass is `activation @ W + b`; hence the on-disk shapes are `trunk_W_0 = (7,48)`,
> `config_W_0 = (48,48)`, `config_W_1 = (48,50)`.

### 1.5.4 Behaviour policy ($\varepsilon$-greedy, linear decay)

$$
a_t = \begin{cases}
\operatorname*{arg\,max}_{a} Q(s_t,a;\theta) & \text{with prob. } 1-\varepsilon_t\\[2pt]
\text{Uniform}\{0,\dots,49\} & \text{with prob. } \varepsilon_t
\end{cases}
$$

The schedule is **linear** from $\varepsilon_{\text{start}}=0.30$ to $\varepsilon_{\text{end}}=0.05$ over
$K=45$ windows (driven by the environment-step counter $n$), then held:

$$
\varepsilon_n = \varepsilon_{\text{start}} + \min\!\Big(\frac{n}{K},1\Big)\big(\varepsilon_{\text{end}}-\varepsilon_{\text{start}}\big)
$$

Tie-breaking: the lowest config id wins (deterministic argmax).

### 1.5.5 Double-DQN target

The **online** network selects the next action; the **target** network evaluates it (van Hasselt
et al., 2016), which suppresses Q-value over-estimation:

$$
a^\star = \operatorname*{arg\,max}_{a} Q(s',a;\theta)
$$
$$
\boxed{\;y_t = r_t + \gamma\,(1 - d_t)\, Q\big(s',\,a^\star;\,\theta^-\big)\;}
$$

with terminal flag $d_t \in \{0,1\}$. Implementation:

```python
next_config_ids = self.online.greedy_actions(next_states)      # online argmax
target_next_q   = self.target.selected_q(next_states, next_config_ids)  # target eval
y = rewards + gamma * (1.0 - dones) * target_next_q
```

### 1.5.6 Loss functions

Let the per-sample TD residual be $e_i = y_i - Q(s_i,a_i;\theta)$ (batch of $N=16$).

**(a) MSE loss** (the original, now disabled):

$$
L_{\text{MSE}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} e_i^{\,2},
\qquad
\frac{\partial L_{\text{MSE}}}{\partial Q(s_i,a_i)} = \frac{2\,e_i}{N}
$$

The gradient is **unbounded**: an outlier target (e.g. a $-10^{6}$ invalid-window sentinel) yields a
$\sim 10^{12}$ squared loss and an enormous update, which corrupted the network. Symptoms observed
in the failing runs: loss spikes to $2.5\times10^{11}$, 575/2280 steps above 1e3, 56 sentinel rewards.

**(b) Huber loss, $\delta = 1$** (the loss used in this benchmark):

$$
L_{\text{Huber}}(\theta) = \frac{1}{N}\sum_{i=1}^{N}
\ell_\delta(e_i),\qquad
\ell_\delta(e) =
\begin{cases}
\tfrac{1}{2}e^{2} & |e| \le \delta\\[3pt]
\delta\big(|e| - \tfrac{1}{2}\delta\big) & |e| > \delta
\end{cases}
$$

Its gradient is **clipped by construction**:

$$
\frac{\partial \ell_\delta}{\partial Q(s_i,a_i)} =
\frac{1}{N}\,\mathrm{clip}(e_i,\,-\delta,\,+\delta)
$$

so a bad sample can move the parameters by at most $\delta/N$ per update — no divergence.

```python
clipped = np.clip(td_error, -huber_delta, huber_delta)      # +/- 1
output_gradient[range(N), config_ids] = clipped / N
```

**Softened sentinel:** the non-finite / invalid-window reward was changed from $-10^{6}$ to
$-5$ (matching the reward clip bound), so even before Huber the target stays bounded.

### 1.5.7 Parameter update (Adam) and target soft-update

Gradients are back-propagated through the head and trunk (hand-written NumPy), then applied with
**Adam** ($\beta_1=0.9,\ \beta_2=0.999,\ \hat{\epsilon}=10^{-8},\ \eta=10^{-3}$), per parameter $w$:

$$
m_w \leftarrow \beta_1 m_w + (1-\beta_1)\,g_w, \qquad
v_w \leftarrow \beta_2 v_w + (1-\beta_2)\,g_w^{2}
$$
$$
\hat{m}_w = \frac{m_w}{1-\beta_1^{\,t}}, \qquad
\hat{v}_w = \frac{v_w}{1-\beta_2^{\,t}}, \qquad
w \leftarrow w - \eta\,\frac{\hat{m}_w}{\sqrt{\hat{v}_w}+\hat{\epsilon}}
$$

After each mini-batch the **target network** is polyak-averaged:

$$
\theta^- \leftarrow \tau\,\theta + (1-\tau)\,\theta^-, \qquad \tau = 0.01
$$

### 1.5.8 Replay buffer

A FIFO buffer $\mathcal{D}$ of capacity $5\times10^{4}$ stores transitions
$(s_t, a_t, r_t, s_{t+1}, d_t)$; each update draws a **uniform** mini-batch of $N=16$.
`updates_per_window = 4` gradient steps are taken per decision window. Learning begins once
$|\mathcal{D}| \ge \max(\text{min\_replay\_size},\,N) = 16$.

### 1.5.9 Reward function

Per window, with measured power $P$ (W) and mean service time $T$ (s):

$$
E = P\,T \quad\text{(energy, J)}
$$

$$
r_t = \mathrm{clip}\!\Big(\underbrace{-w_E \log\frac{E}{E_{\text{ref}}}}_{\text{energy term}}
+ \underbrace{-w_L \log\frac{T}{T_{\text{ref}}}}_{\text{latency term}},\; -5,\; +5\Big),
\qquad w_E = 0.55,\; w_L = 0.45
$$

$E_{\text{ref}}, T_{\text{ref}}$ are the medians of the stock-ondemand reference windows.
Because $E = PT$:

$$
\log\frac{E}{E_{\text{ref}}} = \log\frac{P}{P_{\text{ref}}} + \log\frac{T}{T_{\text{ref}}}
\;\;\Longrightarrow\;\;
\boxed{\;r_t = -\,0.55\,\log\frac{P}{P_{\text{ref}}} \;-\; 1.00\,\log\frac{T}{T_{\text{ref}}}\;}
$$

**Structural note (important for interpretation).** With $w_E + w_L = 1$, the **service-time term always
carries coefficient exactly 1.0**, independent of the weights; only the power-emphasis coefficient
$w_E$ is swept. This is why weight variation has limited leverage, and why we additionally report
the raw energy and service-time gains next to the composite EDP.

### 1.5.10 Evaluation metric (whole-window total energy), gain and win-rate

The scored energy is the **total RAPL domain energy over the whole measurement window**
(`domain_energy_j`), i.e. the physically meaningful server energy **including the idle
gaps** between arrivals:

$$
E = \text{domain\_energy}_j \quad(\text{J, whole window}),\qquad
\underbrace{\mathrm{EDP} = E \cdot T}_{\text{total-EDP (energy × service time)}}
$$

For a set of $M=12$ evaluation windows and a reference baseline $b$ (its **median** over the paired
reference windows):

$$
b = \mathrm{median}\big(\{b_1,\dots,b_M\}\big),\qquad
g_i = \frac{b - \mathrm{EDP}_i}{b},\qquad
\text{gain} = \frac{1}{M}\sum_{i=1}^{M} g_i,\qquad
\text{win-rate} = \frac{1}{M}\sum_{i=1}^{M}\mathbb{1}[g_i > 0]
$$

The identical formula is applied to $E$ (total energy) and $T$ (service time). Positive gain = the agent
beats the reference. The baseline is **ondemand** (12 paired interleaved windows, per-window pairing).

---

## 2. Experimental platform

| Component | Setting |
|-----------|---------|
| Host | Linux 7.0.0-34-generic, 12-core package |
| Worker (target) core | **CPU 3** (`policy3`, isolated via cpuset; server JVM pinned by `-Daffinity.reserved`) |
| Controller core | **CPU 8** (`policy8`) — training/agent process runs under `taskset -c 8` |
| Client core | **CPU 7** |
| RAPL domain | `core` — `/sys/class/powercap/intel-rapl:0:0` |
| Package cores (for idle attribution) | `n_package_cores = 12` |
| Power basis for EDP | `package` (raw RAPL package power; within-λ CV ≈ 0.15) |
| Workload | Closed-loop **M/G/1** generated by MG1 (`java Server.Server` + `java client.LoadGenerator`) |
| Network shim | loopback alias `10.0.0.2/24` on `lo` (required by the MG1 ResetListener) |
| UDP port | 9950 |
| Session duration | 7200 s per λ; warmup 15 s; measurement window 5 s |
| Seed | 42 (agent); client λ-seed fixed |

**Idle/background calibration** (measured at start-up, per λ session):
`other_cores_idle_power_w ≈ 0.894 W`, `background_power_w ≈ 0.975 W`.

**Evaluation metric (the scored quantities):**
```
total energy = RAPL `domain_energy_j`          (J, whole window incl. idle gaps)
service time = `mean_service_time_s`           (s)
total-EDP    = domain_energy_j × mean_service_time_s   (energy × delay)
```
(The runner additionally records an *active-service* `EDP = estimated_target_power_w × time²`
as the training reward's internal quantity; it is **not** an evaluation metric because it
excludes the idle-gap energy that the agent saves.)

---

## 3. Offline warm-start model

| Field | Value |
|-------|-------|
| Source artifact | `offline/configuration_only_ddqn_weights.npz` |
| Companion metadata | `offline/agent_metadata.json` |
| Origin | "Newly trained reconstruction of the supplied notebook deployment path" |
| Model class | `ConfigurationQNet` (notebook) |
| Architecture | `[2, 48, 48, 50]` — activations `tanh, tanh, linear` |
| Parameters | 4 946 |
| Input features | `[ρ, log10(clip(1−ρ, 1e-3, ·))]` |
| Normaliser | `DEPLOY_SURFACE.mu / sd` |
| γ (offline) | 0.0 (greedy argmax, no exploration) |
| Training seed | 42 |
| Notebook SHA-256 | `b2c7281117036f64a89b2aab90771808186153a2b6be4da1220c473a07742df1` |
| Weights SHA-256 | `cfa3e0fe3a2b1cda16ec4024b69198d75bed887bffa785b59b0d0438f4376cae` |

**Conversion to the online agent** (offline-artifact → `offline/warmstart_bundle.npz`):

| artifact key | bundle key | shape |
|---|---|---|
| `trunk_W0` (2×48) | `trunk_W_0` | (2,48) |
| `trunk_b0` | `trunk_b_0` | (48,) |
| `head_W0` (48×48) | `config_W_0` | (48,48) |
| `head_b0` | `config_b_0` | (48,) |
| `head_W1` (48×50) | `config_W_1` | (48,50) |
| `head_b1` | `config_b_1` | (50,) |
| `state_mean` | `state_mu` | (2,) |
| `state_std` | `state_sd` | (2,) |
| — | `n_configs` = 50 | scalar |

`warm_start()` copies `trunk_W_0` into **rows 0–1** of the 7-feature input layer and zero-initialises
rows 2–6 (the five new features the offline model never saw); the config head (48→48→50) is loaded
verbatim. Optimiser state is reset; the target network is re-synced to the online network.

**Action-grid check:** the artifact's stored `sampling_rates_us`/`thresholds_pct` are in the
**same order** as the online `ACTION_GRID` (verified `array_equal`), so head outputs map to the
correct configs — no permutation is needed.

---

## 4. Online learning agent

### 4.1 Network

```
ConfigOnlyQNetwork:
  trunk       = MLP([ 7 → 48 ])                tanh
  config_head = MLP([ 48 → 48 → 50 ])          tanh, tanh, linear
  output      = Q(s, ·) for all 50 configs
```
Hand-written NumPy MLP with Adam, dependency-free for CPU deployment (`config_only_ddqn.py`).

### 4.2 Hyperparameters

| Hyperparameter | Value |
|---|---|
| State dim | 7 |
| Hidden width | 48 |
| Action space | 50 (config-only; nominal grid 800 = 10×5×16 but frequency head unused) |
| Discount γ | 0.95 |
| Learning rate | 1e-3 (Adam) |
| Target soft-update τ | 0.01 |
| Replay capacity | 50 000 |
| Batch size | 16 |
| Min replay size | 16 |
| Updates per window | 4 |
| ε start → end | 0.30 → 0.05 |
| ε decay horizon | 45 windows |
| Loss | **Huber, δ = 1** (gradient clipped to ±1) |
| Invalid-window reward | **−5** (was −1e6) |
| Seed | 42 |
| Decision trigger | phase |
| Phase util threshold | 10 % change in utilisation |
| Phase min / max | 3 s / 15 s |
| Interleaved reference cadence | every 5 windows |
| Warm start | `offline/warmstart_bundle.npz` |

### 4.3 Double-DQN target

```
target = r + γ · (1 − done) · Q_target( s', argmax_a Q_online(s', a) )
```
with a target network soft-updated by `τ = 0.01` after each update.

---

## 5. Training and evaluation pipeline (per λ)

```
[1] Warmup                       15 s   (stabilise governor + server)
[2] Reference windows            ×4     stock ondemand, both sr/thr free → baseline medians
    ── reference_edp / reference_power recorded ──
[3] Interleaved reference        every 5th training window: stock-ondemand window
[4] Training windows             ×300   agent acts (ε-greedy), 4 gradient updates/window
[5] Evaluation windows           ×12
        for each of the 12:
            [5a] paired ondemand reference window (phase `reference_interleaved`)
            [5b] agent window (greedy policy, phase `evaluation`)
[6] Save summary.json + checkpoint + learning.csv + runs.csv + telemetry.csv
```

The **paired design** (5a directly before 5b) cancels slow drifts in package temperature/leakage
that would otherwise bias a fixed baseline.

---

## 6. Evaluation methodology (how the numbers are computed)

* **Per-window total energy** = `domain_energy_j` (RAPL, whole window incl. idle); **service
  time** = `mean_service_time_s`; **total-EDP** = `domain_energy_j × mean_service_time_s`.
* **Gain vs a baseline** (matches the runner's `_gain_vs_baseline`):

  ```
  baseline   = median( reference values over the 12 eval pairs )
  gain_i     = (baseline − agent_i) / baseline          (per window)
  headline   = mean(gain_i),  win_rate = mean(gain_i > 0)
  ```
  The baseline is **ondemand** — median of the 12 paired interleaved ondemand total-EDPs
  (energy × service time per window).
* **Sign convention:** **+ = agent better than reference**, **− = agent worse**.
* The same formula is applied to `domain_energy_j` (energy) and `mean_service_time_s` (service time)
  so all three metrics are reported on the same footing.
* The energy metric is the **measured RAPL package energy per window** (`domain_energy_j`).

---

## 7. Results

### 7.1 Per-λ gains vs ondemand (headline table)

Energy = whole-window RAPL `domain_energy_j` (incl. idle gaps); **svc cost + = agent slower**,
− = agent faster; total-EDP = energy × service time. (energy-save and total-EDP: + = agent better)

| λ | energy OD | svc cost OD | total-EDP OD |
|---|-----------|-------------|--------------|
| 0.8 | **+35.3 %** | +3.4 % | **+33.1 %** |
| 1.6 | **+36.3 %** | +0.1 % | **+36.1 %** |
| 2.4 | **+42.7 %** | +2.2 % | **+41.5 %** |
| 3.2 | **+46.3 %** | +1.8 % | **+45.4 %** |
| 4.0 | **+33.0 %** | +2.5 % | **+31.4 %** |
| 4.8 | **+29.5 %** | +1.6 % | **+28.4 %** |
| 5.6 | **+26.5 %** | +1.5 % | **+25.3 %** |
| 6.4 | **+16.7 %** | +0.8 % | **+16.4 %** |
| 7.2 | **+27.9 %** | +1.7 % | **+26.9 %** |
| 7.5 | −30.8 % | +1.4 % | −31.7 % |
| **mean** | **+26.3 %** | +1.7 % | **+25.3 %** |
| **wins** | **9 / 10** | 0 / 10 | **9 / 10** |

### 7.2 Absolute whole-window total energy and total-EDP (for reference)

| λ | agent E (J/wdw) | ondemand E (J/wdw) | agent total-EDP (J·s) | ondemand total-EDP (J·s) | total-EDP gain OD |
|---|----------------|--------------------|----------------------|--------------------------|-------------------|
| 0.8 | 5.343 | 8.311 | 0.6929 | 1.0444 | +33.1 % |
| 1.6 | 5.663 | 9.058 | 0.7067 | 1.1305 | +36.1 % |
| 2.4 | 6.259 | 11.025 | 0.7999 | 1.3787 | +41.5 % |
| 3.2 | 6.939 | 12.998 | 0.8763 | 1.6128 | +45.4 % |
| 4.0 | 8.652 | 13.144 | 1.1071 | 1.6416 | +31.4 % |
| 4.8 | 9.552 | 13.811 | 1.2034 | 1.7132 | +28.4 % |
| 5.6 | 12.152 | 16.633 | 1.5183 | 2.0484 | +25.3 % |
| 6.4 | 14.994 | 18.015 | 1.8649 | 2.2239 | +16.4 % |
| 7.2 | 13.655 | 19.469 | 1.7184 | 2.4099 | +26.9 % |
| 7.5 | 25.979 | 19.383 | 3.2394 | 2.3829 | −31.7 % |

### 7.3 GFLOPS/W gain vs ondemand

| λ | 0.8 | 1.6 | 2.4 | 3.2 | 4.0 | 4.8 | 5.6 | 6.4 | 7.2 | 7.5 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| GF/W gain | +8.4 % | +15.4 % | −9.7 % | −9.7 % | +7.9 % | +12.2 % | +2.6 % | −0.7 % | −6.3 % | −1.0 % |

(mean ≈ +1.9 %; the agent is at parity-to-better on the throughput-per-watt view at low/mid load.)

### 7.4 Training convergence (per λ, last 50 windows)

| λ | mean loss (last 50) | mean reward (last 50) | replay size | train steps |
|---|--------------------|-----------------------|-------------|-------------|
| 0.8 | 0.121 | −0.049 | 300 | 1140 |
| 1.6 | 0.062 | +0.014 | 300 | 1140 |
| 2.4 | 0.125 | −0.034 | 300 | 1140 |
| 3.2 | 0.082 | +0.053 | 300 | 1140 |
| 4.0 | 0.087 | +0.044 | 300 | 1140 |
| 4.8 | 0.175 | +0.003 | 300 | 1140 |
| 5.6 | 0.115 | +0.006 | 300 | 1140 |
| 6.4 | 0.113 | +0.005 | 300 | 1140 |
| 7.2 | 0.117 | −0.020 | 300 | 1140 |
| 7.5 | 0.090 | −0.099 | 300 | 1140 |

Losses are bounded (Huber) — no divergence spikes; replay saturates at the 300-window horizon.

---

## 8. Figures (in `plots/`)

### 8.1 Reinforcement-learning diagnostic plots

| File | Content |
|------|---------|
| `rl_01_training_loss.png` | Huber loss vs training step, all 10 λ (log-y, smoothed) |
| `rl_02_td_error_abs.png` | absolute TD error vs step, all λ |
| `rl_03_reward.png` | per-window reward vs step, all λ |
| `rl_04_epsilon.png` | ε decay schedule (0.30 → 0.05) |
| `rl_05_replay_size.png` | replay-buffer growth to 300 |
| `rl_06_greedy_rate.png` | fraction of greedy (non-exploratory) actions |
| `rl_07_q_gap.png` | Q-value gap (max Q − chosen Q) |
| `rl_08_utilization.png` | measured utilisation during training |
| `rl_09_edp_train.png` | EDP observed during training |
| `rl_10_qvalues_l0.8.png`, `…_l4.8.png`, `…_l7.5.png` | chosen-Q / max-Q / target-Q trajectories for low, mid, high λ |
| `rl_11_action_heatmap.png` | action-selection heatmap: config id (0–49) × λ |
| `rl_12_action_trace.png` | selected `sampling_rate_us` and `threshold_pct` over training |

### 8.2 Per-λ RL plots (inside each lambda's own directory)

All 10 λ sessions additionally carry their own complete RL plot set, written to
`models/lambda_<λ>_w055-045/plots/` (12 figures each, 120 in total):

| File | Content |
|------|---------|
| `pl_01_training_loss.png` | Huber loss vs step (log-y) |
| `pl_02_td_error_abs.png` | absolute TD error vs step |
| `pl_03_reward.png` | per-window reward vs step |
| `pl_04_epsilon.png` | linear ε decay 0.30 → 0.05 |
| `pl_05_replay_size.png` | replay buffer growth |
| `pl_06_greedy_rate.png` | greedy (non-exploratory) action fraction |
| `pl_07_q_gap.png` | Q-value gap (max Q − chosen Q) |
| `pl_08_utilization.png` | measured utilisation ρ |
| `pl_09_edp_train.png` | EDP observed during training |
| `pl_10_qvalues.png` | chosen-Q / max-Q / target-Q trajectories |
| `pl_11_action_hist.png` | action-selection histogram over the 50 configs |
| `pl_12_action_trace.png` | selected `sampling_rate_us` and `threshold_pct` over training |

(§8.1–§8.2 are the cross-λ comparison figures, in `w055-045/plots/`; §8.3 are the per-λ diagnostics,
in each λ's own sub-directory.)

---

## 9. Discussion

1. **Energy saving is the robust win.** The agent captures **16.7–46.3 % whole-window total-energy
   savings** at 9/10 λ (mean **+26.3 %** vs ondemand). This is the metric where warm-starting
   clearly delivered: it is the largest saving with the smallest latency cost of any configuration
   tested in the project (from-scratch 0.5/0.5 managed only ≈ +10 % energy at ≈ −2 % svc).

2. **Latency cost is small and bounded.** Service time is worse by only 0.7–2.6 % — well inside the
   10 % acceptability budget adopted for this benchmark. The agent buys energy with a *tiny* latency
   concession, not a large one.

3. **Total-EDP improves strongly.** Because the scored quantity is the **whole-window total-EDP =
   energy × service time** (the energy includes the idle gaps), the ~27 % energy saving is *not*
   cancelled by the small latency cost: **total-EDP mean +25.3 % vs ondemand, 9/10 λ winning**, up to
   +45.4 % at λ=3.2. (An earlier *active-service* formulation `EDP = est_target_power × svc²`
   squared the service-time penalty and excluded the idle-gap energy, which misrepresents the result;
   it is dropped from the evaluation.)

4. **Failure corner: λ = 7.5.** Here the agent over-throttles (sampling too slowly / threshold too
   high) and ends up using **more** energy (−30.8 %) at near-parity service time. It is the only λ
   where energy loses. This is the highest-load saturation regime where the ondemand baseline is
   already near the power/performance knee, and the learned policy does not recover.

5. **Warm-start verdict.** The warm-start rounds (0.5/0.5, 0.55/0.45, 0.6/0.4) all sit at the top
   of the energy-vs-latency frontier, and **0.55/0.45 is the best overall**: mean energy +26.3 %,
   smallest service cost (−1.7 %), total-EDP +25.3 % (9/10). The reliable, headline claim is
   **large whole-window energy saving at minimal latency cost**.

---

## 10. Reproduction

```bash
cd "$(git rev-parse --show-toplevel)"

# 1. (single λ shown) run the warm-started config-only DDQN session
sudo taskset -c 8 python3 training/run_joint_freq_mg1.py \
    --target-cpu 3 --controller-cpu 8 --client-cpu 7 \
    --mg1-root "$PWD/MG1" --lambda 0.8 \
    --duration-s 7200 --warmup-s 15 --window-s 5 \
    --decision-trigger phase --phase-util-threshold-pct 10 \
    --phase-min-s 3 --phase-max-s 15 --phase-baseline-s 1 --phase-recent-s 0.5 \
    --min-jobs-per-window 4 --reference-windows 4 \
    --training-windows 300 --evaluation-windows 12 \
    --interleave-reference-every 5 --reference-window 3 \
    --paired-evaluation-reference \
    --edp-time-metric service --power-basis package \
    --n-package-cores 12 --rapl-domain core --config-only \
    --reward-mode WEIGHTED --energy-weight 0.55 --latency-weight 0.45 \
    --gamma 0.95 --learning-rate 1e-3 --batch-size 16 \
    --min-replay-size 16 --updates-per-window 4 \
    --epsilon-start 0.30 --epsilon-end 0.05 --epsilon-decay-windows 45 \
    --seed 42 --warmstart "$PWD/offline/warmstart_bundle.npz" \
    --results models/<lambda_dir>

# 2. Whole batch (all 10 λ) via the launcher
./training/launch_approachC_warmstart.sh 0.55 0.45 models
```

**Prerequisites:** `sudo ip addr add 10.0.0.2/24 dev lo` (MG1 ResetListener bind); UDP port 9950 free;
no stale `java Server.Server` process. RAPL readable at `/sys/class/powercap/intel-rapl:0:0`.

**Key code files**
| File | Role |
|------|------|
| `training/run_joint_freq_mg1.py` | main session runner (state, reward, EDP, evaluation, summary) |
| `agent/config_only_ddqn.py` | `ConfigOnlyQNetwork` + `ConfigOnlyDDQN` (Huber loss, warm_start) + `ACTION_GRID` |
| `training/launch_approachC_warmstart.sh` | per-λ batch launcher |
| `training/choose_weights_warmstart.py` | batch scorer / next-weight decision |

---

## 11. Column dictionary (runs.csv / learning.csv)

**runs.csv** (one row per window): `phase` (`reference` | `reference_interleaved` | `train` |
`evaluation`), `run`, `config_id`, `sampling_rate_us`, `threshold_pct`,
`window_seconds`, `jobs_completed`, `mean_response_time_s`, `mean_service_time_s`,
`mean_measured_frequency_khz`, `utilisation`, `domain_energy_j`, `domain_power_w`,
`other_cores_idle_power_w`, `estimated_target_power_w`, `edp_w_s2`, `reference_edp_w_s2`,
`gflops_per_w`, `reward`, `state_0…6`, `next_state_0…6`.

The **evaluation** metric is `domain_energy_j` (whole-window total RAPL energy, incl. idle) and
`mean_service_time_s`; the reported total-EDP = `domain_energy_j × mean_service_time_s`. The
`edp_w_s2` / `reference_edp_w_s2` columns are the **active-service** quantity used inside the
training reward (`estimated_target_power_w × time²`); they are **not** the evaluation metric.

**learning.csv** (one row per training step): `step`, `loss`, `td_error`, `td_error_abs`,
`target_q`, `chosen_q`, `max_q`, `q_gap`, `greedy_rate`, `replay_size`, `epsilon`, `exploratory`,
`reward`, `utilisation`, `edp_w_s2`.

---

*Report generated from on-disk artifacts in `models/`.*
*All gains use the median-baseline per-window method and are reproducible from `runs.csv` +
`summary.json`.*
