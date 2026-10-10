#!/bin/bash
# Warm-started Approach C launcher: config-only + WEIGHTED reward, starting each
# lambda's network from the offline artifact (warmstart_bundle.npz).
# Usage: launch_approachC_warmstart.sh <energy_weight> <latency_weight> <results_abs_dir>
set -euo pipefail
E_W="${1:?energy weight}"
L_W="${2:?latency weight}"
RESULTS="$(mkdir -p "$3" && cd "$3" && pwd)"


cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
MG1_ROOT=../MG1
WARMSTART="../offline/warmstart_bundle.npz"
LAMBDAS="0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 7.5"
CONTROLLER_CPU=8

mkdir -p "$RESULTS"

if ! ip addr show dev lo | grep -q "10.0.0.2"; then
    sudo ip addr add 10.0.0.2/24 dev lo 2>/dev/null || echo "  [warn] could not add 10.0.0.2 alias"
fi

for LAMBDA in $LAMBDAS; do
    LAM_DIR="$RESULTS/approachC-w${E_W//./}-${L_W//./}-l$(echo "$LAMBDA" | tr -d '.')-$LAMBDA"
    echo ""
    echo "==================================================================="
    echo "  Approach C WARMSTART | λ=$LAMBDA | CONFIG-ONLY + WEIGHTED (${E_W}/${L_W})"
    echo "  warmstart -> $WARMSTART"
    echo "  results   -> $LAM_DIR"
    echo "==================================================================="

    if [ -n "$(find "$LAM_DIR" -name summary.json 2>/dev/null | head -1)" ]; then
        echo "  [skip] $LAM_DIR already ran (summary.json found)"
        continue
    fi
    mkdir -p "$LAM_DIR"

    sudo taskset -c "$CONTROLLER_CPU" env PYTHONPATH="$PWD/../agent" "$PYTHON" run_joint_freq_mg1.py \
        --target-cpu 3 --controller-cpu "$CONTROLLER_CPU" --client-cpu 7 \
        --mg1-root "$MG1_ROOT" \
        --lambda "$LAMBDA" \
        --duration-s 7200.0 --warmup-s 15.0 --window-s 5.0 \
        --decision-trigger phase --phase-util-threshold-pct 10.0 \
        --phase-min-s 3.0 --phase-max-s 15.0 \
        --phase-baseline-s 1.0 --phase-recent-s 0.5 \
        --min-jobs-per-window 4 \
        --reference-windows 4 \
        --training-windows 300 --evaluation-windows 12 \
        --interleave-reference-every 5 --reference-window 3 \
        --paired-evaluation-reference \
        --performance-reference-per-lambda-windows 12 \
        --edp-time-metric service \
        --power-basis package \
        --n-package-cores 12 --rapl-domain core \
        --config-only \
        --reward-mode WEIGHTED \
        --energy-weight "$E_W" --latency-weight "$L_W" \
        --n-frequency-bins 16 \
        --min-frequency-khz 800000 --max-frequency-khz 2100000 \
        --gamma 0.95 --learning-rate 1e-3 --batch-size 16 \
        --min-replay-size 16 --updates-per-window 4 \
        --epsilon-start 0.30 --epsilon-end 0.05 --epsilon-decay-windows 45 \
        --seed 42 \
        --warmstart "$WARMSTART" \
        --results "$LAM_DIR" \
        --session-name "approachC-warmstart-w${E_W//./}-${L_W//./}-l$(echo "$LAMBDA" | tr -d '.')"

    echo "  [done] λ=$LAMBDA"
done

echo ""
echo "═══ ALL 10 LAMBDAS COMPLETE (WARMSTART, weights ${E_W}/${L_W}) ═══"
echo "results -> $RESULTS"