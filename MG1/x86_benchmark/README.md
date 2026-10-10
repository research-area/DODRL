# MG1 x86 Benchmark

x86 benchmark scripts for ondemand governor parameter studies.


## Files

- `x86_benchmark/run_ondemand.sh`: run one static configuration (`sampling_rate`, `threshold`).
- `x86_benchmark/run_threshold_sweep.sh`: run a full grid automatically.
- `x86_benchmark/lib/common.sh`: shared helpers (safe PID handling, output directory creation, archiving).
- `x86_benchmark/helpers.sh`: minimal benchmark math and hardware helper functions used by the runner.
- `x86_benchmark/add_to_excel.py`: local Excel update utility used by this package.

## Output layout

Logs are archived per configuration under:

`data/x86_benchmark/date/sr/<sampling_rate>/threshold/<threshold>/`

Each run directory contains:

- benchmark logs moved from `build/`
- `nohup.log` (full execution log for that run)

## Single configuration run

```bash
cd x86_benchmark
nohup ./run_ondemand.sh \
  --core 3 \
  --runs 10 \
  --min-freq 800000 \
  --max-freq 2100000 \
  --sampling-rate 10000 \
  --threshold 95 &
```

## Full sweep run

```bash
cd x86_benchmark
nohup ./run_threshold_sweep.sh \
  --core 3 \
  --runs 10 \
  --min-freq 800000 \
  --max-freq 2100000 \
  --sampling-rates "10000,20000,30000,40000,50000,60000,70000,80000,90000,100000" \
  --thresholds "75,80,85,90,95" &
```

## Notes

- Requires sudo access for governor changes and low-level metrics.
- Expects Java server/client artifacts in `build/`.
