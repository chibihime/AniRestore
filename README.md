---
title: AniRestore
emoji: 🎨
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: 4x anime cover-art restoration with Real-ESRGAN
---

# AniRestore

CUDA/Triton-accelerated anime cover-art restoration: a pretrained Real-ESRGAN
checkpoint, profiled down to its actual GPU bottleneck, with a custom fused
Triton kernel replacing the slice of PyTorch ops that profiling identified —
plus the measurement work needed to know whether that kernel actually earns
its keep end-to-end, not just at the op level.

Built on cover art re-derived from the same MyAnimeList dataset used by
[AniOntology](https://github.com/chibihime/AniOntology) (same anime IDs across both projects — see
[Relationship to AniOntology](#relationship-to-aniontology)).

## Try it

**Live demo:** [huggingface.co/spaces/prutha-14/AniRestore](https://huggingface.co/spaces/prutha-14/AniRestore)
(free CPU tier, so it runs the PyTorch fallback path rather than the fused
Triton kernel; benchmark numbers below are from a T4 GPU). Pushes to `master`
redeploy it via `.github/workflows/deploy-hf-space.yml`.

To run it yourself:

```bash
python app.py
```

Opens a Gradio UI: upload an anime image (or pick one of the samples in
`examples/`), click **Restore**, see the original and the 4x-restored output
side by side with inference time and device shown. The demo launches with
`share=True`, so Gradio also prints a public `*.gradio.live` link (valid for
72 hours) that anyone can open while the process is running.

## Problem statement

Real-ESRGAN's anime-optimized checkpoint (`RealESRGAN_x4plus_anime_6B`) is a
solid off-the-shelf 4x super-resolution model, but running it as-is leaves an
open question for anyone trying to actually serve it: **where does the time
go, and can a targeted kernel change move the needle?** That's the real
subject of this project — not just "wrap a pretrained model," but profile
it, find the actual bottleneck, build a kernel for it, and rigorously measure
whether it helped.

## Approach

1. **Baseline.** Vendor the RRDBNet architecture directly (not
   `pip install basicsr` — its `setup.py` is broken on current Python) so the
   model's internals are fully editable, load the pretrained checkpoint,
   verify it restores real cover art correctly.
2. **Profile.** `torch.profiler` on a Tesla T4 (Kaggle) found the convolution
   itself dominates (65.9% of GPU time, cuDNN's tuned Winograd kernel — not a
   sane Triton target), but **bias-add and LeakyReLU each ran as separate
   elementwise kernels after nearly every conv** — 11.83% of GPU time
   combined, confirmed by call counts matching the architecture exactly.
   Full writeup: [results/phase3_findings.md](results/phase3_findings.md).
3. **Kernel.** `src/kernels/fused_bias_act.py` fuses those two ops into one
   Triton kernel — one read of the conv output, one write of the activated
   result, instead of two round trips through global memory. Gated behind a
   flag on every model class so the stock and fused paths can run with
   identical weights for direct comparison.
4. **Correctness gate.** 19 tests (`tests/`) — kernel-level (shapes, batch
   sizes, non-contiguous input) and model-level (PSNR/SSIM parity against the
   real checkpoint) — all pass on Tesla T4, several **bit-exact** (PSNR = inf,
   MSE = 0.0). A kernel doesn't ship until this passes, full stop.
5. **Measure, honestly.** The first (naive) benchmark showed *no* wall-clock
   improvement despite the GPU-compute-time win — Triton's own per-launch
   dispatch overhead was absorbing it. Tried CUDA graph capture (the standard
   fix for that exact symptom) — it made things *worse*. Diagnosed that
   comparing separately-invoked benchmark processes was too noisy to trust
   (the *baseline* drifted ~10% across runs with zero code changes). Built an
   interleaved, single-process, paired comparison instead
   (`src/benchmark_compare.py`) — see
   [results/phase4_findings.md](results/phase4_findings.md) for the full
   three-round investigation.

## Benchmark (the honest result)

Tesla T4, Kaggle, 256x256 input, batch size 1, 100 interleaved rounds,
paired against `stock_eager` per round (cancels shared GPU-contention noise
instead of comparing independent run means):

| Config | mean ms/frame | vs. stock (paired) | significant? |
|---|---|---|---|
| Stock PyTorch (eager) | 215.34 | — | — |
| **Triton fused kernel (eager)** | **211.94** | **-3.40ms (-1.58%)** | **yes, t = -10.18** |
| Stock + CUDA graph | 217.15 | +1.82ms (+0.85%, *slower*) | yes, t = +4.78 |
| Triton + CUDA graph | 213.24 | -2.10ms (-0.97%) | yes, t = -4.96 |

**Shipped config: Triton fused kernel, no CUDA graph** — the fastest option,
and the only one with an unconfounded story: a real, statistically
significant **1.58% latency reduction / 1.60% throughput increase**, fully
explained by the 53% GPU-compute-time cut to the targeted op (confirmed via
profiler). CUDA graphs regressed *both* configurations here, most likely
because some of cuDNN's fastest convolution algorithms aren't graph-capture
safe — so they're implemented and tested, but intentionally not used.

The honest headline is not "53% faster" (that's compute time for one op, not
end-to-end latency) or "no effect" (that was a measurement artifact) but a
modest, correctly-attributed **1.6%** real-world win. Full data:
[results/benchmark_compare.json](results/benchmark_compare.json).

## Repo structure

```
src/
  mal/                MAL API client (standalone, reusable — see below)
  model/
    rrdbnet_arch.py    vendored RRDBNet architecture (BSD-3-Clause, Real-ESRGAN)
    restorer.py        Restorer: load checkpoint, run inference, eager/Triton/CUDA-graph
  kernels/
    fused_bias_act.py  the fused Triton kernel
  fetch_cover_manifest.py, download_covers.py   cover-art data pipeline
  data_loader.py       image <-> tensor helpers
  benchmark.py, benchmark_compare.py, profile_baseline.py   measurement scripts
tests/                 kernel + model + CUDA-graph correctness gates
results/               benchmark numbers, profiler tables, findings writeups
examples/              curated sample cover art for the demo
docs/environment_setup.md   Kaggle Notebook setup (this project's GPU environment)
app.py                 the Gradio demo
```

## Running the benchmarks yourself

Requires an NVIDIA GPU (developed on a Kaggle T4 — see
[docs/environment_setup.md](docs/environment_setup.md) for that setup, since
local dev here is Apple Silicon with no CUDA at all):

```bash
python -m pytest tests/ -v
python -m src.benchmark_compare --n 100 --warmup 10 --resize 256
```

## Relationship to AniOntology

The cover-art dataset here isn't re-fetched from scratch — `src/fetch_cover_manifest.py`
reads the anime IDs and cover URLs straight out of
[AniOntology](https://github.com/chibihime/AniOntology)'s already-ingested MAL dataset
(`data/raw/myanimelist_anime.json`), so both projects stay pointed at the
exact same set of anime. `src/mal/` is a standalone MAL API v2 client (no
import dependency on AniOntology) used only to backfill any entries missing a
cached cover URL.

## Licensing

This project's own code is MIT-licensed (see [LICENSE](LICENSE)). That does
**not** cover: the vendored RRDBNet architecture (BSD-3-Clause, from
Real-ESRGAN), the pretrained checkpoint (subject to the original project's
license, not redistributed here), or the anime cover-art images in
`examples/`/`data/` (copyrighted material used for personal, non-commercial
research/demonstration purposes only). See [LICENSE](LICENSE) for details
before reusing anything beyond the code itself.
