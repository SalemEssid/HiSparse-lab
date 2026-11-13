# Phase 1 measurements (G1)

Run on 2026-11-12, host `srv-vgpu-02`, commit `7e4738c`, scripts in `hw/`. Raw data: `hw/results/*.json`.
Everything in this file is a measurement (fact) unless marked as derived.

## Machine

| Item | Value |
| --- | --- |
| GPU | NVIDIA A40-24Q: time-sliced 24 GB vGPU profile on a 48 GB A40 (sm 8.6, 84 SMs), 23.79 GiB usable |
| Driver | 535.230.02 (CUDA 12.2); guest driver fixed by the vGPU host |
| Software | torch 2.6.0+cu124, transformers 4.57.6, Python 3.11 |
| PCIe link | hidden by vGPU (`nvidia-smi` reports N/A) |
| Host RAM | 62.8 GiB total, 57.2 GiB available |
| Pinned memory | 32 GiB allocated without error (probe limit; memlock ulimit 8229012 KB does not apply) |
| CPU | Intel Xeon Gold 6348 @ 2.60 GHz, 32 logical cores visible |
| SDPA backends | flash, efficient, cudnn, math all usable |
| Not supported | CUDA virtual memory management (`expandable_segments` fails: "operation not supported") |

## Contiguous copies

Device GB/s from CUDA events; wall-clock GB/s (launch + sync included) in parentheses.

| Size | H2D pinned | H2D pageable | D2H pinned | D2H pageable |
| --- | --- | --- | --- | --- |
| 4 KiB | 0.33 (0.19) | 0.28 (0.17) | 0.38 (0.21) | 0.29 (0.18) |
| 64 KiB | 4.54 (2.54) | 2.68 (1.96) | 6.04 (3.30) | 3.37 (2.30) |
| 256 KiB | 15.97 (9.44) | 7.20 (5.74) | 18.04 (11.10) | 5.40 (4.55) |
| 1 MiB | 22.19 (18.32) | 9.11 (8.40) | 23.51 (19.52) | 6.64 (6.26) |
| 4 MiB | 23.26 (22.00) | 13.72 (13.24) | 25.03 (23.63) | 8.80 (8.61) |
| 16 MiB | 24.25 (23.86) | 16.26 (16.10) | 25.57 (25.11) | 10.00 (9.93) |
| 64 MiB | 22.82 (22.58) | 10.85 (10.83) | 26.31 (26.18) | 9.86 (9.85) |
| 256 MiB | 25.28 (25.25) | 9.79 (9.78) | 26.37 (26.33) | 9.84 (9.83) |

## Scattered gathers (miss-fetch model)

Random units from a 4 GiB pinned pool into GPU slots, wall clock. `loop` = one copy per unit;
`stage` = CPU `index_select` into a pinned buffer, then one copy.
Units: 4 KiB = one token of one layer (K+V, bf16); 64 KiB = one 16-token page of one layer.

| Units | 4 KiB loop µs/unit (GB/s) | 4 KiB stage µs/unit (GB/s) | 64 KiB loop µs/unit (GB/s) | 64 KiB stage µs/unit (GB/s) |
| --- | --- | --- | --- | --- |
| 1 | 29.01 (0.14) | 35.39 (0.12) | 28.11 (2.33) | 45.35 (1.45) |
| 4 | 14.41 (0.28) | 9.42 (0.43) | 13.13 (4.99) | 19.72 (3.32) |
| 16 | 10.81 (0.38) | 2.89 (1.42) | 9.71 (6.75) | 14.45 (4.54) |
| 64 | 9.98 (0.41) | 0.93 (4.42) | 8.85 (7.40) | 13.25 (4.95) |
| 128 | 9.65 (0.42) | 0.64 (6.44) | 8.67 (7.56) | 12.91 (5.08) |
| 256 | 9.67 (0.42) | 0.50 (8.25) | 8.51 (7.70) | 12.71 (5.16) |
| 512 | 9.54 (0.43) | 0.35 (11.57) | 8.56 (7.65) | 12.73 (5.15) |

## Dense decode baseline: Qwen3-4B, full KV on GPU

HF transformers eager mode, SDPA, bf16, `DynamicCache`, greedy, 128 decode steps, needle at 50% depth.
Config matches the project notes: 36 layers, 8 KV heads, head dim 128, `max_position_embeddings` 40960.
Weights 7.55 GiB; KV 144 KiB per token, 64 KiB per 16-token page per layer.

| Prompt tokens | Prefill (s) | TPOT median / p90 (ms) | KV after prefill (GiB) | Peak prefill / decode (GiB) | Needle |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 0.985 | 30.91 / 31.08 | 1.164 | 9.28 / 8.73 | found |
| 16,384 | 2.381 | 31.89 / 32.08 | 2.328 | 11.01 / 9.91 | found |
| 32,000 | 6.341 | 43.40 / 43.45 | 4.548 | 14.32 / 12.16 | found |

## Derived numbers for later phases

- **Link speed for planning:** 25 GB/s H2D pinned; one 64 KiB page at that speed takes 2.6 µs.
- **Per-copy cost in PyTorch:** about 8.5–10 µs per copy call (64+ units) at 4 KiB and at 64 KiB alike, so a per-page fetch loop is limited by copy submissions, not bytes.
- **KV accounting is exact:** memory after prefill = analytic KV + the last chunk's hidden states (32,000 × 2,560 × 2 B = 0.153 GiB); 4.395 + 0.153 = 4.548 GiB.
- **Host KV capacity:** 57.2 GiB available / 144 KiB per token ≈ 416K tokens, about 13 requests at 32K before leaving headroom for the process and OS.
