"""Interleaved, single-process comparison of all four Restorer configurations.

Separate `python -m src.benchmark` process invocations showed the *baseline*
drifting ~10% across consecutive runs with zero code changes (see
results/phase4_findings.md) -- evidence of shared/noisy Kaggle GPU conditions,
not a real effect. Running all four configs in one process and alternating
between them round-by-round means every config sees the same ambient noise
at the same moments, so the *relative* (paired) comparison is trustworthy
even though absolute latency still drifts over the run. Uses torch.cuda.Event
for GPU-side timing instead of time.perf_counter(), removing CPU-side
scheduling jitter from the measurement itself.

Usage:
    python -m src.benchmark_compare --input data/raw_covers --n 100 --warmup 10 \
        --resize 256 --out results/benchmark_compare.json
"""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

import torch

from src.data_loader import load_image_tensor
from src.model.restorer import Restorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIGS = [
    ("stock_eager", dict(use_triton_kernel=False, use_cuda_graph=False)),
    ("triton_eager", dict(use_triton_kernel=True, use_cuda_graph=False)),
    ("stock_graph", dict(use_triton_kernel=False, use_cuda_graph=True)),
    ("triton_graph", dict(use_triton_kernel=True, use_cuda_graph=True)),
]
BASELINE_NAME = "stock_eager"


def _timed_restore_ms(restorer: Restorer, img: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    restorer.restore(img)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _summarize(xs: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "stdev_ms": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "p95_ms": sorted(xs)[int(0.95 * len(xs))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_covers")
    parser.add_argument("--n", type=int, default=100, help="number of interleaved rounds")
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup passes per config")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--out", default="results/benchmark_compare.json")
    parser.add_argument("--seed", type=int, default=0, help="shuffles per-round config order")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this comparison requires CUDA")

    paths = sorted(
        p for p in Path(args.input).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if len(paths) < args.n + args.warmup:
        raise ValueError(f"need at least {args.n + args.warmup} images, found {len(paths)}")

    images = [load_image_tensor(p, resize_to=args.resize) for p in paths[: args.n + args.warmup]]
    warmup_imgs, timed_imgs = images[: args.warmup], images[args.warmup :]

    logger.info("building and warming up %d configs (each captures/autotunes independently)", len(CONFIGS))
    restorers = {}
    for name, kwargs in CONFIGS:
        r = Restorer(**kwargs)
        for img in warmup_imgs:
            r.restore(img)
        torch.cuda.synchronize()
        restorers[name] = r
        logger.info("  %s ready", name)

    rng = random.Random(args.seed)
    latencies = {name: [] for name, _ in CONFIGS}

    logger.info("running %d interleaved rounds (config order shuffled each round)", len(timed_imgs))
    for i, img in enumerate(timed_imgs):
        order = [name for name, _ in CONFIGS]
        rng.shuffle(order)
        for name in order:
            latencies[name].append(_timed_restore_ms(restorers[name], img))
        if (i + 1) % 25 == 0:
            logger.info("  round %d/%d", i + 1, len(timed_imgs))

    results = {name: _summarize(xs) for name, xs in latencies.items()}

    baseline_xs = latencies[BASELINE_NAME]
    for name, xs in latencies.items():
        if name == BASELINE_NAME:
            continue
        # paired per-round difference -- cancels out noise common to a round
        # (e.g. transient GPU contention), unlike comparing separate means.
        diffs = [b - a for a, b in zip(baseline_xs, xs)]
        results[name]["paired_diff_vs_baseline_ms"] = {
            "mean": statistics.mean(diffs),
            "stdev": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        }

    results["_meta"] = {
        "gpu_name": torch.cuda.get_device_name(0),
        "n_rounds": len(timed_imgs),
        "n_warmup": len(warmup_imgs),
        "resize": args.resize,
        "seed": args.seed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("=== summary (mean ms/frame, GPU-event timing) ===")
    for name, _ in CONFIGS:
        r = results[name]
        extra = ""
        if "paired_diff_vs_baseline_ms" in r:
            d = r["paired_diff_vs_baseline_ms"]
            extra = f"  paired diff vs {BASELINE_NAME}: {d['mean']:+.2f}ms (stdev {d['stdev']:.2f}ms)"
        logger.info(
            "%-14s mean=%.2fms median=%.2fms p95=%.2fms%s", name, r["mean_ms"], r["median_ms"], r["p95_ms"], extra
        )

    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
