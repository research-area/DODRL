#!/bin/bash

set -euo pipefail

# Applies governor and frequency bounds for a selected CPU set.
configure_cpu_performance() {
  local cpu="$1"
  local governor="$2"
  local min_frequency="$3"
  local max_frequency="$4"

  if [[ "$governor" == "userspace" ]]; then
    echo "setting cpu: $cpu frequency: $min_frequency governor: userspace"
    sudo cpupower --cpu "$cpu" frequency-set -g userspace -d "$min_frequency" -u "$max_frequency"
    sudo cpufreq-set --cpu "$cpu" -f "$min_frequency"
    return
  fi

  if [[ "$governor" == "ondemand" ]]; then
    echo "setting cpu: $cpu frequency range: $min_frequency -> $max_frequency governor: ondemand"
    sudo cpupower --cpu "$cpu" frequency-set -g ondemand -d "$min_frequency" -u "$max_frequency"
    return
  fi

  echo "Unsupported governor: $governor" >&2
  return 1
}

# Returns lambda sweep step and max lambda based on max CPU frequency.
increase_lamda() {
  case "$1" in
    2100000) echo "0.86 8.6" ;;
    2000000) echo "0.82 7.42" ;;
    1900000) echo "0.75 6.76" ;;
    1800000) echo "0.71 6.36" ;;
    1700000) echo "0.67 6.06" ;;
    1600000) echo "0.62 5.60" ;;
    1500000) echo "0.58 5.25" ;;
    1400000) echo "0.55 4.93" ;;
    1300000) echo "0.51 4.54" ;;
    1200000) echo "0.49 4.43" ;;
    1100000) echo "0.43 3.90" ;;
    1000000) echo "0.38 3.42" ;;
    900000)  echo "0.35 3.19" ;;
    800000)  echo "0.32 2.86" ;;
    *)
      echo "unknown frequency"
      return 1
      ;;
  esac
}

# Reads per-core voltage using MSR 0x198 (Intel only).
read_cpu_voltage_msr() {
  local core="$1"
  local raw=""

  raw=$(sudo rdmsr -p "$core" 0x198 -u --bitfield 47:32 2>/dev/null || true)
  if [[ -z "$raw" ]]; then
    echo "N/A"
    return
  fi

  local voltage
  voltage=$(echo "scale=3; $raw / 8192" | bc -l)
  if awk "BEGIN {exit !($voltage > 0)}"; then
    echo "$voltage"
  else
    echo "0"
  fi
}

# Piecewise-linear model for whole-CPU idle power as a function of frequency.
interpolate_cpu_idle_power() {
  local freq_hz="$1"
  freq_hz=$(printf "%.0f" "$freq_hz")

  awk -v f="$freq_hz" '
  BEGIN {
    fq[1]=800000; pw[1]=7.14;
    fq[2]=1450000; pw[2]=7.35;
    fq[3]=2100000; pw[3]=7.49;
    n=3;

    if (f <= fq[1]) { printf "%.6f", pw[1]; exit }
    if (f >= fq[n]) { printf "%.6f", pw[n]; exit }

    for (i=1; i<n; i++) {
      if (f >= fq[i] && f <= fq[i+1]) {
        power = pw[i] + (f - fq[i]) * (pw[i+1] - pw[i]) / (fq[i+1] - fq[i]);
        printf "%.6f", power;
        exit
      }
    }

    printf "%.6f", pw[1]
  }'
}

# Dynamic power model for whole CPU: P = Ceff * V^2 * f.
compute_cpu_p_dynamic() {
  local freq_khz="$1"
  local volt="$2"
  local ceff="0.0000172773"

  freq_khz=$(printf "%.0f" "$freq_khz")
  if [[ -z "$freq_khz" || -z "$volt" ]]; then
    echo "0"
    return
  fi

  bc -l <<< "scale=10; $ceff * $volt^2 * $freq_khz"
}

# Core-level modeled power derived from whole-CPU idle/dynamic components.
modeled_power() {
  local core_freq="$1"
  local core_volt="$2"
  local core_util="$3"
  local number_of_cores="$4"

  local cpu_idle_power
  local cpu_dynamic_power
  cpu_idle_power=$(interpolate_cpu_idle_power "$core_freq")
  cpu_dynamic_power=$(compute_cpu_p_dynamic "$core_freq" "$core_volt")

  local core_idle_power
  local core_dynamic_power
  core_idle_power=$(echo "$cpu_idle_power / $number_of_cores" | bc -l)
  core_dynamic_power=$(echo "$cpu_dynamic_power / $number_of_cores" | bc -l)

  echo "$(echo "$core_idle_power + ($core_util / 100) * $core_dynamic_power" | bc -l)"
}
