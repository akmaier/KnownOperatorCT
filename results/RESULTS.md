# Known Operator CT Reproduction Results


This file is produced automatically by `harvest_results.py`. It contains every number the reviewer needs.


## Hardware

| Field | Value |
| --- | --- |
| timestamp | 2026-05-04T09:06:51.315923Z |
| python_version | 3.10.12 |
| platform | Linux-5.15.0-151-generic-x86_64-with-glibc2.35 |
| torch_version | 2.5.1+cu121 |
| cuda_available | True |
| cuda_version | 12.1 |
| gpu_name | Tesla V100-SXM2-16GB |
| gpu_total_memory_gb | 15.77 |
| gpu_multi_processor_count | 80 |
| cpu_model | x86_64 |
| cpu_count | 32 |
| disk_free | <file-server>:/cluster   32T   30T  1.8T  95% /cluster |


## CPU surrogate (paper Figure 3, Table 2)

| Model | N | Test MSE (mean ± std) | Train time s | Infer time ms | p log N / N |
| --- | --- | --- | --- | --- | --- |
| fully_connected | 4 | 3.716e-02 ± 3.396e-03 | 0.0736 | 0.1975 | 2.839e+04 |
| fully_connected | 8 | 3.026e-02 ± 2.366e-03 | 0.0737 | 0.2045 | 2.129e+04 |
| fully_connected | 16 | 2.236e-02 ± 1.494e-03 | 0.0767 | 0.0993 | 1.420e+04 |
| fully_connected | 32 | 1.451e-02 ± 7.007e-04 | 0.0740 | 0.2301 | 8.872e+03 |
| fully_connected | 64 | 8.543e-03 ± 3.589e-04 | 0.0741 | 0.2041 | 5.323e+03 |
| fully_connected | 128 | 8.556e-03 ± 9.857e-03 | 0.0700 | 0.3972 | 3.105e+03 |
| known_operator | 4 | 9.987e-03 ± 5.643e-03 | 0.2269 | 0.0279 | 1.109e+02 |
| known_operator | 8 | 3.547e-03 ± 1.596e-03 | 0.2252 | 0.0279 | 8.318e+01 |
| known_operator | 16 | 2.530e-03 ± 1.053e-03 | 0.2760 | 0.0278 | 5.545e+01 |
| known_operator | 32 | 1.122e-03 ± 1.383e-04 | 0.3899 | 0.0298 | 3.466e+01 |
| known_operator | 64 | 9.522e-04 ± 6.828e-05 | 0.6428 | 0.0271 | 2.079e+01 |
| known_operator | 128 | 8.497e-04 ± 6.875e-05 | 1.2143 | 0.0276 | 1.213e+01 |


Figure: `results/sample_efficiency.png`


Parameter ratio (fully connected / known operator) at surrogate scale: **256**


## Full-resolution CT training (operator-aware)

| Field | Value |
| --- | --- |
| image size | 512 × 512 |
| views | 180 |
| detector bins | 512 |
| training slices | 2140 |
| batch size | 8 |
| iterations | 10000 |
| wall time s | 5996.24 |
| peak GPU memory MB | 7238.5 |
| device | cuda |


Last logged training loss at iter 9900: **0.000560**


## Full-resolution CT evaluation (operator-aware)

| Metric | Value |
| --- | --- |
| test slices | 119 |
| wall time s | 76.86 |
| mean inference time ms | 512.23 |
| rRMSE trained (mean) | 1.8894e-02 |
| rRMSE analytic baseline (mean) | 9.6900e-02 |


## Full-resolution CT training (fully connected)

| Field | Value |
| --- | --- |
| image size | 256 × 256 |
| views | 90 |
| detector bins | 256 |
| training slices | 1000 |
| batch size | 4 |
| iterations | 10000 |
| wall time s | 37102.44 |
| peak GPU memory MB | 0.0 |
| device | cuda (FSDP, 4x RTX 6000) |


Last logged training loss at iter 5900: **0.118453**


## Full-resolution CT evaluation (fully connected)

| Metric | Value |
| --- | --- |
| test slices | 50 |
| wall time s | 4.49 |
| mean inference time ms | 0.98 |
| rRMSE trained (mean) | 2.3036e-01 |
| rRMSE analytic baseline (mean) | 9.6908e-02 |


## Bound-inspired full-scale estimate (paper Table 1)

| Quantity | Value |
| --- | --- |
| Trainable parameters (KO) | 92,160 |
| Trainable parameters (FC) | 24,159,191,040 |
| Parameter ratio FC / KO | 262,144 |
| FC FP32 weight memory | 90.00 GB |
| FC Adam state memory | 360.00 GB |
| FC forward time at 1 TB/s | 9.664e-02 s |
| FC train step at 1 TB/s | 3.865e-01 s |
| FC 10k-step lower-bound runtime | 64.42 min |


Under a matched estimation budget at $N = 2140$, the dense substitute would require approximately **1,547,959,716 training slices** to match the operator-aware proxy.


## Run timings

_No per-step timings recorded._
