# Phase 4: Fused Triton Kernel — Findings

**Hardware:** Kaggle Notebook, Tesla T4 (Turing, sm_75), torch 2.10.0+cu128, Triton 3.6.0
**Model:** RealESRGAN_x4plus_anime_6B (RRDBNet, 6 RRDB blocks)

## What was built

`src/kernels/fused_bias_act.py` fuses each conv's bias-add and LeakyReLU into
one Triton kernel, replacing the two separate elementwise kernel launches
Phase 3 identified as 11.83% of GPU time combined (`results/phase3_findings.md`).
Gated behind `use_triton_kernel` on `ResidualDenseBlock`/`RRDB`/`RRDBNet`/`Restorer`
so stock and fused paths can run with identical weights for direct comparison.

## Correctness

19 tests in `tests/` (kernel-level shape/batch-size/non-contiguous-input cases,
model-level PSNR/SSIM parity with the real checkpoint, and CUDA-graph parity)
all pass on Tesla T4. Several configurations are **bit-exact** (PSNR = inf,
MSE = 0.0) between the stock and fused paths — this is an elementwise op with
no reduction/associativity concerns, so exact equality is the expected result,
not a coincidence.

## Performance: three rounds of measurement

**Round 1 (naive, separate `python -m src.benchmark` process per config):**
Kernel fusion cut GPU compute time for the targeted op by 53% (435ms -> 204ms
of self-CUDA time across 20 profiled passes), confirmed via `torch.profiler`.
But wall-clock latency was unchanged (185.3ms baseline vs 185.7-185.9ms
Triton, twice) — a new `cuLaunchKernelEx` line (1800 calls, matching our
kernel's call count 1:1) showed Triton's own dispatch overhead was absorbing
the entire saving. At batch_size=1 with ~1800 small kernel launches per
frame, launch overhead and compute time are comparable in magnitude.

**Round 2 (CUDA graph capture, still separate processes):** CUDA graph replay
is the standard fix for exactly this symptom (eliminates per-op Python/driver
dispatch by replaying a recorded kernel sequence). Result was the opposite of
expected: both stock+graph (+9.2%) and Triton+graph (+10.1%) were *slower*
than eager baseline. Worse, the baseline itself drifted 185.3ms -> 197.0ms ->
204.4ms across three separate-process runs with **no code change to that
path at all** — clear evidence that comparing separately-invoked processes on
a shared/free-tier Kaggle GPU is too noisy to trust for effects this small.
(`cudnn.benchmark=True` was also tried in isolation and didn't resolve the
drift, consistent with the drift being environmental rather than an algorithm-
selection artifact alone.)

**Round 3 (interleaved, single-process, paired comparison, GPU-event timing):**
`src/benchmark_compare.py` builds all four configs once, alternates between
them in a shuffled order every round (so every config sees the same ambient
noise at the same moment), and reports the *paired* per-round difference
against `stock_eager` rather than comparing independent means. This is the
trustworthy result:

| Config | mean ms/frame | paired diff vs stock_eager | t-statistic |
|---|---|---|---|
| stock_eager (baseline) | 215.34 | — | — |
| **triton_eager** | **211.94** | **-3.40ms (-1.58%)** | **-10.18** |
| stock_graph | 217.15 | +1.82ms (+0.85%) | +4.78 |
| triton_graph | 213.24 | -2.10ms (-0.97%) | -4.96 |

All three diffs are statistically significant (|t| >> 2), because the paired
design cancels out the shared per-round noise that made Rounds 1-2
uninterpretable. Full data: `results/benchmark_compare.json`.

## Conclusion and shipped configuration

- **Ship `use_triton_kernel=True`, `use_cuda_graph=False`.** This is the
  fastest configuration and the only one with a clean, unconfounded
  performance story: a real (if modest) **1.58% latency reduction / 1.60%
  throughput increase**, fully explained by the 53% GPU-compute-time cut to
  the targeted bias+LeakyReLU epilogue.
- **CUDA graphs are not worth using here.** Both graphed configurations are
  slower than their eager counterparts (stock_graph is the single slowest
  config measured, even slower than plain stock_eager) — likely because some
  of cuDNN's fastest convolution algorithms rely on workspace-allocation
  patterns that aren't graph-capture-safe, so cuDNN falls back to a more
  conservative algorithm inside the captured region. Triton+graph is faster
  than stock+graph, but still slower than Triton alone, so graph capture adds
  no value on top of the kernel fusion on this hardware/workload.
- **Methodology matters as much as the kernel itself here.** The honest
  headline is not "53% faster" (that's GPU compute time for one op, not
  end-to-end latency) nor "no effect" (that was an artifact of noisy
  separate-process measurement) but a modest, rigorously-isolated **1.6%**
  end-to-end win — small, but real, reproducible, and correctly attributed.
