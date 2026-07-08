"""Run baseline Real-ESRGAN inference over a folder of cover-art images.

Usage:
    python -m src.run_baseline_inference \
        --input data/raw_covers --output data/restored --limit 10
"""

import argparse
import logging
from pathlib import Path

from src.data_loader import load_image_tensor, tensor_to_pil
from src.model.restorer import Restorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_covers")
    parser.add_argument("--output", default="data/restored")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if args.limit:
        paths = paths[: args.limit]

    kwargs = {"checkpoint_path": args.checkpoint} if args.checkpoint else {}
    restorer = Restorer(**kwargs)
    logger.info("running on device: %s", restorer.device)

    for path in paths:
        img = load_image_tensor(path)
        restored = restorer.restore(img)
        out_path = out_dir / f"{path.stem}.png"
        tensor_to_pil(restored).save(out_path)
        logger.info("%s: %s -> %s", path.stem, tuple(img.shape), tuple(restored.shape))

    logger.info("restored %d images into %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
