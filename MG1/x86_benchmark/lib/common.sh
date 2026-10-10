#!/bin/bash

set -euo pipefail

# Canonical paths shared by benchmark scripts.
COMMON_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$COMMON_LIB_DIR/.." && pwd)"
ROOT_DIR="$(cd "$BENCHMARK_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DATA_ROOT="$ROOT_DIR/data/x86_benchmark"

# Force-kill helper used during cleanup to avoid leaked samplers.
safe_kill() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# Safe wait wrapper: ignores wait errors for already-finished processes.
safe_wait() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]]; then
    wait "$pid" 2>/dev/null || true
  fi
}

# Verifies required binaries before a benchmark starts.
require_commands() {
  local missing=0
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Missing required command: $cmd" >&2
      missing=1
    fi
  done
  if [[ $missing -ne 0 ]]; then
    return 1
  fi
}

# Creates the canonical result path: date/sr/<sampling_rate>/threshold/<threshold>.
ensure_output_dir() {
  local sampling_rate="$1"
  local threshold="$2"
  local date_tag="$3"

  local out_dir="$DATA_ROOT/${date_tag}/sr/${sampling_rate}/threshold/${threshold}"
  mkdir -p "$out_dir"
  printf "%s\n" "$out_dir"
}

# Ensures server IP alias exists before client-server benchmark traffic starts.
ensure_server_ip() {
  local iface="$1"
  if ! ip addr show dev "$iface" | grep -q "10.0.0.2/24"; then
    sudo ip addr add 10.0.0.2/24 dev "$iface" || true
  fi
}

# Moves files matching a glob only when matches exist.
archive_pattern_if_exists() {
  local pattern="$1"
  local destination="$2"

  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob

  if [[ ${#matches[@]} -gt 0 ]]; then
    mv "${matches[@]}" "$destination/"
  fi
}

# Collects and archives all artifacts produced by one configuration run.
archive_configuration_logs() {
  local core="$1"
  local sampling_rate="$2"
  local threshold="$3"
  local destination="$4"

  pushd "$BUILD_DIR" >/dev/null

  archive_pattern_if_exists "core_frequency_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "cpu_power_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "cpu_cores_frequency_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "cpu_cores_utilization_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "cpu_cores_voltage_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "core_utilization_*_${sampling_rate}_${threshold}_*.log" "$destination"
  archive_pattern_if_exists "prediction_log_*" "$destination"
  archive_pattern_if_exists "Repeat.csv" "$destination"
  archive_pattern_if_exists "SteadyStateProbability.txt" "$destination"
  archive_pattern_if_exists "workbook.xlsx" "$destination"

  popd >/dev/null
}
