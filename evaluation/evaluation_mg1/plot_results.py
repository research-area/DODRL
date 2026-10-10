#!/usr/bin/env python3
"""Generate the paper's M/G/1 evaluation plots from existing runs.csv files.

Metrics (matching results.md):
    - energy per job (E/job)
    - FLOPS/Watt (total_flops / domain_energy_j)
    - service time (svc)
    - agent gains vs ondemand (E/job, whole E, svc)

Each metric is drawn as a line plot and as a grouped bar plot.

Usage:
    python3 plot_results.py [--results evaluation_mg1/results]
                            [--output evaluation_mg1/plots]

Outputs (PNG files in the output directory):
    energy_per_job.png / energy_per_job_bars.png
    flops_per_watt.png  / flops_per_watt_bars.png
    svc.png             / svc_bars.png
    gains.png           / gains_bars.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAMBDAS = [0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 7.5]
_HERE = Path(__file__).resolve().parent
LOAD_PCT = {lam: 10 * (i + 1) for i, lam in enumerate(LAMBDAS)}   # 10% .. 100%
COLORS = {"ondemand": "#1f77b4", "agent": "#2ca02c"}
MARKERS = {"ondemand": "o", "agent": "^"}
MODES = ("ondemand", "agent")

XLAB = "CPU load level (%)"


def load_x(lams):
    """Map λ values to CPU load level percentages (10% .. 100%)."""
    return [LOAD_PCT[l] for l in lams]


def style_x_axis(ax, show_xlabel=True):
    ticks = list(range(10, 101, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    if show_xlabel:
        ax.set_xlabel(XLAB)


def bar_x_axis(ax, lams, fontsize=8):
    """x-axis for grouped bars: tick labels are CPU load levels."""
    ticks = load_x(lams)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=fontsize)


def load_runs(results_dir: Path, lam: float):
    lam_tag = f"lam{str(lam).replace('.', '_')}"
    path = results_dir / lam_tag / "runs.csv"
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'nan')]
    return float(np.mean(v)) if v else float('nan')


def gather(results_dir: Path):
    data = {m: {"lam": [], "E": [], "EJ": [], "svc": [], "jobs": [], "P": [], "FW": []}
            for m in MODES}
    for lam in LAMBDAS:
        rows = load_runs(results_dir, lam)
        if rows is None:
            continue
        modes = {}
        for r in rows:
            modes.setdefault(r["mode"], []).append(r)
        for m in MODES:
            if m not in modes:
                continue
            rs = modes[m]
            data[m]["lam"].append(lam)
            data[m]["E"].append(mean(rs, "domain_energy_j"))
            data[m]["EJ"].append(mean(rs, "energy_per_job_j"))
            data[m]["svc"].append(mean(rs, "mean_service_time_s") * 1000)
            data[m]["jobs"].append(mean(rs, "jobs_completed"))
            data[m]["P"].append(mean(rs, "power_w"))
            data[m]["FW"].append(mean(rs, "total_flops") / mean(rs, "domain_energy_j") / 1e6)
    return data


def plot_metric(data, key, ylabel, title, path, yscale="linear"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODES:
        ax.plot(load_x(data[m]["lam"]), data[m][key], marker=MARKERS[m],
                color=COLORS[m], label=m, linewidth=2, markersize=7)
    style_x_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale == "log":
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_bars(data, key, ylabel, title, path, yscale="linear"):
    """Grouped bars: one group per CPU load level, one bar per mode."""
    fig, ax = plt.subplots(figsize=(9, 5))
    lams = data["agent"]["lam"]
    pos = load_x(lams)
    width = 3.5
    for m in MODES:
        vals = []
        for lam in lams:
            if lam in data[m]["lam"]:
                vals.append(data[m][key][data[m]["lam"].index(lam)])
            else:
                vals.append(float('nan'))
        offset = -width / 2 if m == "ondemand" else width / 2
        ax.bar([p + offset for p in pos], vals, width=width, color=COLORS[m],
               edgecolor="black", linewidth=0.6, label=m)
    bar_x_axis(ax, lams, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    if yscale == "log":
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def _gains_series(data, key):
    """Per-λ agent gains vs ondemand for one metric (+ = agent better)."""
    gains = []
    for i, lam in enumerate(data["agent"]["lam"]):
        if lam not in data["ondemand"]["lam"]:
            gains.append(float('nan'))
            continue
        j = data["ondemand"]["lam"].index(lam)
        ag_val = data["agent"][key][i]
        base_val = data["ondemand"][key][j]
        gains.append(100 * (1 - ag_val / base_val))
    return gains


def plot_gains(data, path, kind="line"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("EJ", "E/job gain (%)"), ("E", "Whole E gain (%)"), ("svc", "svc change (%)")]
    lams = data["agent"]["lam"]
    pos = load_x(lams)
    width = 3.5
    for ax, (key, label) in zip(axes, metrics):
        gains = _gains_series(data, key)
        if kind == "line":
            ax.plot(pos, gains, marker="o", color=COLORS["agent"],
                    label="vs ondemand", linewidth=2, markersize=7)
        else:
            colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gains]
            ax.bar(pos, gains, width=width, color=colors,
                   edgecolor="black", linewidth=0.6)
            bar_x_axis(ax, lams)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if kind == "line":
            style_x_axis(ax)
            ax.legend()
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    kind_label = "bar comparison" if kind == "bar" else "line"
    fig.suptitle(f"Agent gains vs ondemand (+ = agent better, {kind_label})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Generate M/G/1 evaluation plots")
    ap.add_argument("--results", type=Path, default=_HERE / "results")
    ap.add_argument("--output", type=Path, default=_HERE / "plots")
    args = ap.parse_args()

    results_dir = args.results.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = gather(results_dir)
    print(f"Generating plots in {output_dir}:")

    plot_metric(data, "EJ", "E/job (J)", "Energy per job",
                output_dir / "energy_per_job.png")
    plot_bars(data, "EJ", "E/job (J)", "Energy per job",
              output_dir / "energy_per_job_bars.png")

    plot_metric(data, "FW", "FLOPS/Watt (MFLOPS/W)",
                "Energy efficiency (useful work per Joule)",
                output_dir / "flops_per_watt.png")
    plot_bars(data, "FW", "FLOPS/Watt (MFLOPS/W)",
              "Energy efficiency (useful work per Joule)",
              output_dir / "flops_per_watt_bars.png")

    plot_metric(data, "svc", "svc (ms)", "Service time", output_dir / "svc.png")
    plot_bars(data, "svc", "svc (ms)", "Service time", output_dir / "svc_bars.png")

    plot_gains(data, output_dir / "gains.png", kind="line")
    plot_gains(data, output_dir / "gains_bars.png", kind="bar")

    print("Done.")


if __name__ == "__main__":
    main()
