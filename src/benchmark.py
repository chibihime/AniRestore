"""Measure baseline Real-ESRGAN inference latency and throughput.

Times single-image inference (batch size 1, matching how the Gradio demo
will call it) over a sample of real cover-art images, with warmup runs
excluded so JIT/cuDNN autotune/kernel-compilation overhead doesn't skew the
numbers. Uses torch.cuda.synchronize() around timing since CUDA kernel
launches are async -- without it wall-clock time would just measure launch
overhead, not actual GPU compute time.

Usage:
    python -m src.benchmark --input data/raw_covers --n 100 --warmup 10 \
        --resize 256 --out results/baseline_benchmark.json
    python -m src.benchmark --triton --out results/triton_benchmark.json
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from src.data_loader import load_image_tensor
from src.model.restorer import Restorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_covers")
    parser.add_argument("--n", type=int, default=100, help="number of timed runs")
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup runs")
    parser.add_argument("--resize", type=int, default=256, help="resize images to this square size")
    parser.add_argument("--out", default="results/baseline_benchmark.json")
    parser.add_argument("--triton", action="store_true", help="use the fused Triton kernel instead of stock ops")
    args = parser.parse_args()

    paths = sorted(
        p for p in Path(args.input).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if len(paths) < args.n + args.warmup:
        raise ValueError(f"need at least {args.n + args.warmup} images, found {len(paths)}")

    restorer = Restorer(use_triton_kernel=args.triton)
    device = restorer.device
    logger.info("device: %s, use_triton_kernel: %s", device, args.triton)

    images = [load_image_tensor(p, resize_to=args.resize) for p in paths[: args.n + args.warmup]]

    warmup_imgs, timed_imgs = images[: args.warmup], images[args.warmup :]

    logger.info("running %d warmup passes (untimed)", len(warmup_imgs))
    for img in warmup_imgs:
        restorer.restore(img)
    _sync(device)

    logger.info("running %d timed passes", len(timed_imgs))
    latencies_ms = []
    for img in timed_imgs:
        _sync(device)
        t0 = time.perf_counter()
        restorer.restore(img)
        _sync(device)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    stdev_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    throughput_fps = 1000.0 / mean_ms

    results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "use_triton_kernel": args.triton,
        "num_timed_runs": len(latencies_ms),
        "num_warmup_runs": len(warmup_imgs),
        "input_resize": args.resize,
        "batch_size": 1,
        "mean_ms_per_frame": mean_ms,
        "median_ms_per_frame": median_ms,
        "p95_ms_per_frame": p95_ms,
        "stdev_ms": stdev_ms,
        "throughput_fps": throughput_fps,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info(
        "mean=%.2fms median=%.2fms p95=%.2fms throughput=%.2f fps -> %s",
        mean_ms,
        median_ms,
        p95_ms,
        throughput_fps,
        out_path,
    )


if __name__ == "__main__":
    main()
