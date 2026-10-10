#!/bin/bash

set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$RUNNER_DIR/.." && pwd)"
ADD_TO_EXCEL_SCRIPT="$RUNNER_DIR/add_to_excel.py"

source "$RUNNER_DIR/helpers.sh"
source "$RUNNER_DIR/lib/common.sh"

# Default runtime configuration (can be overridden by env vars or CLI options).
CORE="${CORE:-3}"
RUNS="${RUNS:-1}"
MIN_FREQ="${MIN_FREQ:-800000}"
MAX_FREQ="${MAX_FREQ:-2100000}"
SAMPLING_RATE="${SAMPLING_RATE:-10000}"
THRESHOLD="${THRESHOLD:-95}"
NUM_CPUS="${NUM_CPUS:-$(( $(nproc) + 1 ))}" # isolated cores are not catched by nproc add +1 to include it
CORE_IDLE_POWER="${CORE_IDLE_POWER:-0.58}" #core idle power at min frequency
GOVERNOR="ondemand"
POWERSTAT_INTERVAL="${POWERSTAT_INTERVAL:-1}"
DURATION_SEC="${DURATION_SEC:-60}"
CLIENT_DURATION_MS="${CLIENT_DURATION_MS:-60000}" #600000
CLIENT_REPEAT="${CLIENT_REPEAT:-1000000}"
WARMUP_SEC="${WARMUP_SEC:-7}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}" #60
SERVER_IFACE="${SERVER_IFACE:-enp0s31f6}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --core N                 Target core (default: $CORE)
  --runs N                 Repetitions per lambda (default: $RUNS)
  --min-freq HZ            Minimum frequency in kHz (default: $MIN_FREQ)
  --max-freq HZ            Maximum frequency in kHz (default: $MAX_FREQ)
  --sampling-rate US       Ondemand sampling_rate in microseconds (default: $SAMPLING_RATE)
  --threshold PERCENT      Ondemand up_threshold (default: $THRESHOLD)
  --num-cpus N             Number of logical CPUs (default: $NUM_CPUS)
  --iface NAME             Network interface to set 10.0.0.2/24 (default: $SERVER_IFACE)
  --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --core) CORE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --min-freq) MIN_FREQ="$2"; shift 2 ;;
    --max-freq) MAX_FREQ="$2"; shift 2 ;;
    --sampling-rate) SAMPLING_RATE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --num-cpus) NUM_CPUS="$2"; shift 2 ;;
    --iface) SERVER_IFACE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_commands bc awk mpstat taskset python3 java sudo cpupower ip || exit 1

# Validate local dependencies early so long runs do not fail late.
if [[ ! -f "$ADD_TO_EXCEL_SCRIPT" ]]; then
  echo "[ERROR] add_to_excel script not found: $ADD_TO_EXCEL_SCRIPT" >&2
  exit 1
fi
BENCHMARK_DATE="${BENCHMARK_DATE:-$(date +%Y%m%d)}"
OUTPUT_DIR="$(ensure_output_dir "$SAMPLING_RATE" "$THRESHOLD" "$BENCHMARK_DATE")"
RUN_LOG="$OUTPUT_DIR/nohup.log"
# Mirror stdout/stderr into a persistent run log.
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Configuration: core=$CORE runs=$RUNS min_freq=$MIN_FREQ max_freq=$MAX_FREQ sampling_rate=$SAMPLING_RATE threshold=$THRESHOLD"

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    safe_kill "$pid"
  done
}
# Always clean background samplers/processes on exit or interruption.
trap cleanup EXIT

echo "[INFO] Preparing CPU governor state"
# Set intel_pstate to passive to allow manual frequency control via cpupower.
echo passive | sudo tee /sys/devices/system/cpu/intel_pstate/status >/dev/null
# Set all cores to userspace governor at min frequency as a known baseline before starting the sweep.
sudo cpupower -c all frequency-set -g userspace >/dev/null
sudo cpupower -c all frequency-set -d "$MIN_FREQ" -u "$MIN_FREQ" >/dev/null
# Set target core to ondemand governor with specified min/max frequencies.
configure_cpu_performance "$CORE" "$GOVERNOR" "$MIN_FREQ" "$MAX_FREQ"

# Read lambda step depending on target core max freqeuency to ensure same utilization levels.
read lambda_step lambda_max <<< "$(increase_lamda "$MAX_FREQ")"
if [[ ! "$lambda_step" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ ! "$lambda_max" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "[ERROR] increase_lamda returned invalid values for max_freq=$MAX_FREQ" >&2
  exit 1
fi

# Sweep lambda values derived from max frequency calibration.
for lambda in $(seq -f "%.2f" "$lambda_step" "$lambda_step" "$lambda_max"); do
  util=$(echo "scale=2; ($lambda / $lambda_max) * 100" | bc)
  # Empirical guardrail: this specific point overloads and causes unstable queue behavior.
  if (( $(echo "$lambda == $lambda_max" | bc -l) )); then
    lambda=$(echo "$lambda_max - 0.5" | bc -l)
  fi

  echo "[INFO] Starting lambda=$lambda (target util=$util%)"

  pkill -9 java 2>/dev/null || true

  for ((i = 1; i <= RUNS; i++)); do
    # Apply ondemand control knobs for the current experiment point.
    echo "$SAMPLING_RATE" | sudo tee /sys/devices/system/cpu/cpufreq/ondemand/sampling_rate >/dev/null
    echo "$THRESHOLD" | sudo tee /sys/devices/system/cpu/cpufreq/ondemand/up_threshold >/dev/null

    echo "[INFO] Iteration $i/$RUNS"
    ensure_server_ip "$SERVER_IFACE"

    pushd "$ROOT_DIR/build" >/dev/null
    export CLASSPATH="../Server/Server/lib/*:."
    # Log file names with clear schema 
    cpu_power_log="cpu_power_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"
    core_freq_log="core_frequency_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"
    cpu_cores_freq_log="cpu_cores_frequency_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"
    cpu_cores_util_log="cpu_cores_utilization_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"
    core_util_log="core_utilization_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"
    cpu_cores_voltage_log="cpu_cores_voltage_${util}_${SAMPLING_RATE}_${THRESHOLD}_${i}.log"

    # Measured package power (powerstat) for this iteration.
    sudo powerstat -R "$POWERSTAT_INTERVAL" "$DURATION_SEC" > "$cpu_power_log" 2>&1 &
    POWERSTAT_PID=$!
    cleanup_pids+=("$POWERSTAT_PID")

    (
      while true; do
        # High-rate frequency snapshots: target core and all cores.
        echo "$(date +%s%N), $(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq)" >> "$core_freq_log"
        freqs="$(date +%s%N)"
        for cpu in $(seq 0 $((NUM_CPUS - 1))); do
          freq=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
          freqs="${freqs},${freq}"
        done
        echo "$freqs" >> "$cpu_cores_freq_log"
        sleep 0.1
      done
    ) &
    MANUAL_FREQ_PID=$!
    cleanup_pids+=("$MANUAL_FREQ_PID")
    # Server running
    java Server.Server "$lambda" "$DURATION_SEC"&
    JAVA_PID=$!
    # Client workload: fixed duration/repeat to keep runs comparable.
    taskset -c 8 java client.LoadGenerator "$lambda" "$CLIENT_DURATION_MS" "$CLIENT_REPEAT" >/dev/null 2>&1 &
    CLIENT_PID=$!

    sleep "$WARMUP_SEC"
    # High-rate utilization snapshots: target core and all cores.
    mpstat -P ALL 1 >> "$cpu_cores_util_log" &
    MPSTAT_ALL_PID=$!
    cleanup_pids+=("$MPSTAT_ALL_PID")

    mpstat -P "$CORE" 1 >> "$core_util_log" &
    MPSTAT_CORE_PID=$!
    cleanup_pids+=("$MPSTAT_CORE_PID")

    # High-rate voltage snapshots: all cores.
    (
      while true; do
        timestamp=$(date +%s.%N)
        line="$timestamp"
        voltage_sum=0
        voltage_count=0

        for cpu in $(seq 0 $((NUM_CPUS - 1))); do
          voltage=$(read_cpu_voltage_msr "$cpu")
          if [[ "$voltage" =~ ^\. ]]; then
            voltage="0$voltage"
          fi
          line="${line},${voltage}"
          if [[ "$voltage" != "N/A" ]] && [[ "$voltage" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            voltage_sum=$(echo "$voltage_sum + $voltage" | bc)
            voltage_count=$((voltage_count + 1))
          fi
        done

        avg_voltage="N/A"
        if [[ $voltage_count -gt 0 ]]; then
          avg_voltage=$(echo "scale=3; $voltage_sum / $voltage_count" | bc)
        fi

        echo "${line},${avg_voltage}" >> "$cpu_cores_voltage_log"
        sleep 0.1
      done
    ) &
    VOLTAGE_PID=$!
    cleanup_pids+=("$VOLTAGE_PID")

    #wait for client and server to finish before killing samplers and post-processing logs
    safe_wait "$JAVA_PID"
    safe_wait "$CLIENT_PID"

    safe_kill "$POWERSTAT_PID"
    safe_kill "$MANUAL_FREQ_PID"
    safe_kill "$MPSTAT_CORE_PID"
    safe_kill "$MPSTAT_ALL_PID"
    safe_kill "$VOLTAGE_PID"

    # Post-process logs to extract average frequency, voltage, utilization and power for the target core.

    read power_count power_energy_sum <<< "$(awk '/^[0-9]{2}:[0-9]{2}:[0-9]{2}/ {sum += $NF; count++} END {print count+0, sum+0}' "$cpu_power_log")"
    if [[ "$power_count" -eq 0 ]]; then
      # Avoid divide-by-zero when powerstat returns no samples.
      power_count=1
    fi

    # Approximate target-core measured power by subtracting idle power of the other cores.
    power_avg_cpu=$(echo "$power_energy_sum / $power_count" | bc -l)
    power_avg_core=$(echo "$power_avg_cpu - 11 * $CORE_IDLE_POWER" | bc -l)

    avg_core_freq=$(awk -F',' '{sum+=$2} END {if (NR > 0) print sum/NR; else print 0}' "$core_freq_log")
    
    core_voltage_col=$((CORE + 1))
    avg_core_voltage=$(awk -F',' -v col="$core_voltage_col" '{total+=$col; count++} END {if (count > 0) print total/count; else print 0}' "$cpu_cores_voltage_log")
    # mpstat column convention used here: take %usr/%system aggregate column.
    utilization=$(awk '/^[0-9]/ && $3 ~ /^[0-9]/ {sum += $4; count++} END {if (count > 0) printf "%.2f", sum/count; else print "0.00"}' "$core_util_log")

    # Modeled core power from frequency, voltage and measured utilization.
    avg_core_modeled_power=$(modeled_power "$avg_core_freq" "$avg_core_voltage" "$utilization" "$NUM_CPUS")

    # Excel schema mapping (kept stable for downstream analysis scripts).
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 1 "$util"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 2 "$SAMPLING_RATE"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 3 "$THRESHOLD"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 14 "$utilization"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 17 "$avg_core_freq"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 18 "$avg_core_modeled_power"
    python3 "$ADD_TO_EXCEL_SCRIPT" workbook.xlsx 19 "$power_avg_core"

    pkill -9 mpstat 2>/dev/null || true
    popd >/dev/null

    echo "[INFO] Iteration $i completed. Sleeping ${SLEEP_BETWEEN_RUNS}s"
    
    #sleep "$SLEEP_BETWEEN_RUNS"
  done
done

echo "[INFO] Archiving logs for sampling_rate=$SAMPLING_RATE threshold=$THRESHOLD"
# Move generated logs from build/ to the benchmark output folder.
archive_configuration_logs "$CORE" "$SAMPLING_RATE" "$THRESHOLD" "$OUTPUT_DIR"

# Restore CPU governor to a known state after the benchmark.
configure_cpu_performance all ondemand $MAX_FREQ $MAX_FREQ

echo "[INFO] Benchmark completed successfully"
echo "[INFO] Logs stored in: $OUTPUT_DIR"
