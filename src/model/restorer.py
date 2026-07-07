"""Reusable Real-ESRGAN (anime 6B) inference wrapper.

Loads RRDBNet + the pretrained checkpoint once, then restores images via
`restore()` / `restore_batch()`. Kept separate from any CLI/script so later
phases (benchmarking, Gradio demo) can import it directly instead of
shelling out to a script.
"""

from pathlib import Path

import torch

from src.model.rrdbnet_arch import RRDBNet

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "RealESRGAN_x4plus_anime_6B.pth"


class Restorer:
    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def restore(self, image: torch.Tensor) -> torch.Tensor:
        """image: CHW float32 tensor in [0, 1]. Returns CHW float32 in [0, 1], 4x upscaled."""
        return self.restore_batch(image.unsqueeze(0))[0]

    @torch.no_grad()
    def restore_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """batch: NCHW float32 tensor in [0, 1]. Returns NCHW float32 in [0, 1], 4x upscaled."""
        batch = batch.to(self.device)
        out = self.model(batch)
        return out.clamp(0, 1).cpu()
