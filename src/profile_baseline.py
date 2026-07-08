"""Profile baseline Real-ESRGAN inference with torch.profiler.

Produces an op-level breakdown (which ops dominate runtime) and a Chrome
trace for visual inspection. For a kernel-level timeline on top of this,
run under `nsys` separately (Linux + NVIDIA only, e.g. on Kaggle):

    nsys profile -o results/baseline_nsys --force-overwrite true \
        python -m src.profile_baseline --n 20

Usage:
    python -m src.profile_baseline --input data/raw_covers --n 20 --resize 256 \
        --out-dir results
    python -m src.profile_baseline --triton --out-dir results
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from src.data_loader import load_image_tensor
from src.model.restorer import Restorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_covers")
    parser.add_argument("--n", type=int, default=20, help="number of profiled forward passes")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--triton", action="store_true", help="use the fused Triton kernel instead of stock ops")
    args = parser.parse_args()

    paths = sorted(
        p for p in Path(args.input).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )[: args.n + args.warmup]

    restorer = Restorer(use_triton_kernel=args.triton)
    device = restorer.device
    logger.info("device: %s, use_triton_kernel: %s", device, args.triton)

    images = [load_image_tensor(p, resize_to=args.resize) for p in paths]
    warmup_imgs, profiled_imgs = images[: args.warmup], images[args.warmup :]

    for img in warmup_imgs:
        restorer.restore(img)
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for img in profiled_imgs:
            restorer.restore(img)
            prof.step()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit)
    print(table)

    prefix = "triton" if args.triton else "baseline"
    table_path = out_dir / f"{prefix}_profile_table.txt"
    table_path.write_text(table, encoding="utf-8")

    trace_path = out_dir / f"{prefix}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    logger.info("wrote op table to %s and chrome trace to %s", table_path, trace_path)
    logger.info("open the trace at chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
