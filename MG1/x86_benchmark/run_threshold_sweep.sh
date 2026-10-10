#!/bin/bash

set -euo pipefail

# Runner script for one (sampling_rate, threshold) configuration.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_ondemand.sh"

CORE="${CORE:-3}"
RUNS="${RUNS:-1}"
MIN_FREQ="${MIN_FREQ:-800000}"
MAX_FREQ="${MAX_FREQ:-2100000}"

# Benchmark configurations.
SAMPLING_RATES=( 10000) #20000 30000 40000 50000 60000 70000 80000 90000 100000)
THRESHOLDS=(95) # 85 90 95)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --core N
  --runs N
  --min-freq HZ
  --max-freq HZ
  --sampling-rates "10000,20000,..."
  --thresholds "75,80,..."
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --core) CORE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --min-freq) MIN_FREQ="$2"; shift 2 ;;
    --max-freq) MAX_FREQ="$2"; shift 2 ;;
    --sampling-rates)
      IFS=',' read -r -a SAMPLING_RATES <<< "$2"
      shift 2
      ;;
    --thresholds)
      IFS=',' read -r -a THRESHOLDS <<< "$2"
      shift 2
      ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Fail fast if the single-run benchmark script cannot be executed.
if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not executable: $RUNNER" >&2
  exit 1
fi

echo "[INFO] Starting sweep"
echo "[INFO] core=$CORE runs=$RUNS min_freq=$MIN_FREQ max_freq=$MAX_FREQ"

# Full factorial sweep across sampling_rate x threshold.
for sampling_rate in "${SAMPLING_RATES[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    echo "[INFO] Running sampling_rate=$sampling_rate threshold=$threshold"

    "$RUNNER" \
      --core "$CORE" \
      --runs "$RUNS" \
      --min-freq "$MIN_FREQ" \
      --max-freq "$MAX_FREQ" \
      --sampling-rate "$sampling_rate" \
      --threshold "$threshold"

    echo "[INFO] Finished sampling_rate=$sampling_rate threshold=$threshold"
  done
done

echo "[INFO] Sweep completed"
