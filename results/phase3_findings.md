# Phase 3: Baseline Benchmark + Profiling Findings

**Hardware:** Kaggle Notebook, Tesla T4 (Turing, sm_75), torch 2.10.0+cu128, Triton 3.6.0
**Model:** RealESRGAN_x4plus_anime_6B (RRDBNet, 6 RRDB blocks, num_feat=64, num_grow_ch=32)
**Input:** 256x256 RGB, batch_size=1 (single-image inference, matching the planned Gradio demo)

## Baseline latency/throughput

From `results/baseline_benchmark.json` (100 timed runs, 10 warmup runs excluded):

| Metric | Value |
|---|---|
| Mean latency | 185.30 ms/frame |
| Median latency | 184.95 ms/frame |
| p95 latency | 187.53 ms/frame |
| Stdev | 1.31 ms |
| Throughput | 5.40 frames/sec |

Low stdev (1.31ms, <1% of mean) indicates a stable, compute-bound workload with
no significant warmup or scheduling noise once warmup passes are excluded.

## Profiler breakdown (torch.profiler, CUDA activities, 20 profiled passes)

Full table: `results/baseline_profile_table.txt`. Self-CUDA-time-sorted, top
contributors:

| Op | Self CUDA % | Self CUDA time | # calls | Per-call avg |
|---|---|---|---|---|
| `aten::cudnn_convolution` (winograd kernel) | 65.9% | 2.557s | 1900 | 1.346ms |
| `aten::cat` | 11.85% | 459.8ms | 1440 | 319us |
| `aten::add_` (in-place, bias add) | 6.61% | 256.7ms | 1920 | 134us |
| `aten::leaky_relu_` | 5.22% | 202.7ms | 1500 | 135us |
| `aten::copy_` (final DtoH transfer) | 4.59% | 178.0ms | 40 | 4.45ms |
| `aten::add` (residual scaling, out-of-place) | 2.68% | 104.0ms | 500 | 208us |
| `aten::mul` (0.2 residual scale) | 1.75% | 68.0ms | 480 | 142us |
| `aten::upsample_nearest2d` | 1.09% | 42.2ms | 40 | 1.06ms |

Call counts line up exactly with RRDBNet's structure (verified, not just
inferred): 96 convs/forward pass x 20 passes = 1920 conv calls; 4 `cat`s per
ResidualDenseBlock x 3 RDBs/RRDB x 6 RRDB blocks x 20 passes = 1440; 1920
`add_` calls is a **1:1 match with conv call count** -- every convolution's
bias is applied as a separate elementwise kernel launch, not fused into the
cuDNN winograd call.

## Bottleneck hypothesis: confirmed, with one refinement

The Phase 3 hypothesis (fused conv+bias+LeakyReLU pattern inside RRDB blocks
as the Triton target) is **confirmed by the data, with an important
refinement**:

- The convolution itself (65.9%) is by far the single largest cost and is
  already running via cuDNN's tuned Winograd kernel -- **not** the right
  target for a from-scratch Triton kernel, since matching cuDNN's hand-tuned
  Winograd implementation is a much larger undertaking than this project's
  scope, and a naive Triton conv would very likely be slower.
- The real opportunity is the **bias-add + LeakyReLU epilogue that currently
  runs as two separate elementwise kernel launches immediately after nearly
  every conv** (`add_`: 6.61%, `leaky_relu_`: 5.22% -- together **11.83%** of
  total GPU time, second only to the convolution itself). Each of these reads
  the conv output from global memory, does trivial elementwise work, and
  writes it back -- twice, when it could be done once. This is exactly the
  "fused conv+bias+LeakyReLU" pattern named in the roadmap.
- A secondary, smaller opportunity is `aten::cat` (11.85%) -- the dense-block
  concatenation of growing feature maps in `ResidualDenseBlock`. This is
  comparable in size to the bias+LeakyReLU opportunity but is a data-movement
  pattern (concatenation) rather than a fusable compute epilogue, and is not
  the hypothesis named in the roadmap -- noted here as a candidate for a later
  kernel if time permits, not the Phase 4 target.

## Phase 4 target

Write a Triton kernel that fuses **bias-add + LeakyReLU** into a single
elementwise pass over each convolution's output, replacing PyTorch's default
`conv2d(bias=True)` -> separate `add_` -> separate `leaky_relu_` chain used
throughout `ResidualDenseBlock` (`src/model/rrdbnet_arch.py`). This targets
the 11.83% of GPU time spent on two elementwise kernel launches that could be
one, without attempting to reimplement the convolution itself.

Expected upper bound on speedup from this fusion alone: roughly halving the
~11.83% (since two kernel launches + two memory round-trips collapse to one),
i.e. a best-case ~5-6% reduction in total latency -- worth doing as the
named, scoped Phase 4 deliverable, but correctness parity (PSNR/SSIM) matters
more here than raw speedup given the modest ceiling.
