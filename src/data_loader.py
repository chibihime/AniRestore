"""Load downloaded cover-art images into tensors for Real-ESRGAN input.

Real-ESRGAN is fully convolutional, so images don't need a fixed input size
for single-image inference — but batched inference/benchmarking needs
same-shaped tensors, hence the optional resize_to.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a CHW float32 tensor normalized to [0, 1]."""
    img = img.convert("RGB")
    arr = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).contiguous()
    return arr.float() / 255.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Inverse of pil_to_tensor: CHW float32 in [0, 1] -> PIL image."""
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def load_image_tensor(path: Path, resize_to: int | None = None) -> torch.Tensor:
    """Load one image as a CHW float32 tensor normalized to [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize_to is not None:
            img = img.resize((resize_to, resize_to), Image.BICUBIC)
        return pil_to_tensor(img)


class CoverArtDataset(Dataset):
    """Dataset over a folder of downloaded cover-art images.

    Set resize_to when you need uniform tensor shapes for batching
    (e.g. throughput benchmarking); leave it None to get each image at its
    native resolution (for single-image inference / visual spot-checks).
    """

    def __init__(self, image_dir: Path | str, resize_to: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.resize_to = resize_to
        self.paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if not self.paths:
            raise ValueError(f"no images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        mal_id = path.stem
        return load_image_tensor(path, resize_to=self.resize_to), mal_id
