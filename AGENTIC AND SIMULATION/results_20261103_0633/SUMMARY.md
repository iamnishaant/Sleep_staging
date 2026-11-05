# Edge simulation results

## check_x86_64

```
{
 "arch": "x86_64",
 "python": "3.12.16",
 "cpus_visible": 4,
 "cgroup_cpu_max": "400000 100000",
 "cgroup_memory_max": "17179869184",
 "ram_available_gb": 11.51,
 "llama_version": [
  "built with GNU 12.2.0 for Linux x86_64"
 ],
 "llama_ok": true,
 "release_pins_ok": true,
 "release_manifest": "5489b3f0c14274f7",
 "network": {
  "tcp_1.1.1.1_53": "blocked (OSError)",
  "dns_pypi.org": "blocked (gaierror)"
 },
 "network_off": true,
 "secret_hits": [],
 "no_secrets": true
}
```

## patha_x86_64

```
{
 "nights": 31,
 "failed_runs": 0,
 "wall_s_median": 151.3,
 "wall_s_p95": 159.8,
 "peak_gb_max": 5.16,
 "peak_gb_median": 5.16,
 "VERIFIED_REPORT": 31,
 "FALLBACK_SUMMARY": 0,
 "REVIEW_ONLY": 0,
 "REJECTED": 0,
 "text_identical_to_reference": 31
}
```

## pathb_llama_x86_64

```
{
 "nights": 3,
 "failed_runs": 0,
 "wall_s_median": 158.3,
 "wall_s_p95": 210.68,
 "peak_gb_max": 5.64,
 "peak_gb_median": 5.63,
 "VERIFIED_REPORT": 2,
 "FALLBACK_SUMMARY": 0,
 "REVIEW_ONLY": 1,
 "REJECTED": 0,
 "parity_nights": 3,
 "parity_arrays_compared": 5,
 "parity_max_abs_diff": 8.344650268554688e-07,
 "parity_hypnograms_identical": 5,
 "E1_pass": true,
 "outcomes_matching_reference": 3
}
```

## pathb_oracle_aarch64

```
{
 "nights": 8,
 "failed_runs": 0,
 "wall_s_median": 133.15,
 "wall_s_p95": 135.9,
 "peak_gb_max": 1.02,
 "peak_gb_median": 1.005,
 "VERIFIED_REPORT": 7,
 "FALLBACK_SUMMARY": 0,
 "REVIEW_ONLY": 1,
 "REJECTED": 0,
 "parity_nights": 8,
 "parity_arrays_compared": 15,
 "parity_max_abs_diff": 7.748603820800781e-07,
 "parity_hypnograms_identical": 15,
 "E1_pass": true
}
```

## pathb_oracle_x86_64

```
{
 "nights": 31,
 "failed_runs": 0,
 "wall_s_median": 18.0,
 "wall_s_p95": 20.5,
 "peak_gb_max": 0.82,
 "peak_gb_median": 0.74,
 "VERIFIED_REPORT": 30,
 "FALLBACK_SUMMARY": 0,
 "REVIEW_ONLY": 1,
 "REJECTED": 0,
 "parity_nights": 31,
 "parity_arrays_compared": 61,
 "parity_max_abs_diff": 1.800060272216797e-05,
 "parity_hypnograms_identical": 61,
 "E1_pass": true
}
```
