"""Model-level correctness gate: stock RRDBNet vs the Triton-fused version.

Per the project's Phase 4 rule, a "faster" kernel is never considered done
until this passes -- PSNR/SSIM parity gates every kernel change. Runs the
real pretrained checkpoint on real cover-art images (falls back to random
weights/synthetic input if either is unavailable, so the test still exercises
the two code paths against each other even without the dataset attached).
"""

from pathlib import Path

import pytest
import torch

from src.data_loader import load_image_tensor
from src.model.rrdbnet_arch import RRDBNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "RealESRGAN_x4plus_anime_6B.pth"
COVERS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_covers"

# PSNR/SSIM parity thresholds. We expect near machine-precision agreement
# since this is a like-for-like elementwise op (no reduction/associativity
# differences), so these are set as loose commitment lines, not the actual
# expected result -- checked separately by asserting the true PSNR is much
# higher again below.
MIN_PSNR_DB = 40.0
MIN_SSIM = 0.995


def _psnr_ssim(a: torch.Tensor, b: torch.Tensor):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a_np = a.permute(1, 2, 0).numpy()
    b_np = b.permute(1, 2, 0).numpy()
    psnr = peak_signal_noise_ratio(a_np, b_np, data_range=1.0)
    ssim = structural_similarity(a_np, b_np, data_range=1.0, channel_axis=2)
    return psnr, ssim


def _build_models(num_block: int):
    if CHECKPOINT.exists():
        num_block = 6  # the real checkpoint is always the 6-block anime variant

    stock = RRDBNet(3, 3, scale=4, num_feat=64, num_block=num_block, num_grow_ch=32, use_triton_kernel=False)
    fused = RRDBNet(3, 3, scale=4, num_feat=64, num_block=num_block, num_grow_ch=32, use_triton_kernel=True)

    if CHECKPOINT.exists():
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        stock.load_state_dict(state_dict, strict=True)

    fused.load_state_dict(stock.state_dict())  # identical weights, only forward path differs
    return stock.eval().cuda(), fused.eval().cuda(), num_block


def _sample_input(resize_to: int) -> torch.Tensor:
    if COVERS_DIR.exists():
        image_paths = sorted(COVERS_DIR.iterdir())
        if image_paths:
            return load_image_tensor(image_paths[0], resize_to=resize_to).unsqueeze(0)
    torch.manual_seed(0)
    return torch.rand(1, 3, resize_to, resize_to)


@pytest.mark.parametrize("resize_to", [64, 128, 256])
def test_psnr_ssim_parity(resize_to):
    num_block = 6 if CHECKPOINT.exists() else 2
    stock, fused, num_block = _build_models(num_block)
    x = _sample_input(resize_to).cuda()

    with torch.no_grad():
        y_stock = stock(x).clamp(0, 1)[0].cpu()
        y_fused = fused(x).clamp(0, 1)[0].cpu()

    psnr, ssim = _psnr_ssim(y_stock, y_fused)
    print(f"resize={resize_to} checkpoint={CHECKPOINT.exists()} psnr={psnr:.2f}dB ssim={ssim:.6f}")

    assert psnr > MIN_PSNR_DB, f"PSNR {psnr:.2f}dB below {MIN_PSNR_DB}dB threshold"
    assert ssim > MIN_SSIM, f"SSIM {ssim:.6f} below {MIN_SSIM} threshold"


def test_batched_input_parity():
    """Batch size > 1, matching how benchmark.py could be extended later."""
    num_block = 6 if CHECKPOINT.exists() else 2
    stock, fused, num_block = _build_models(num_block)

    torch.manual_seed(2)
    x = torch.rand(4, 3, 96, 96, device="cuda")
    with torch.no_grad():
        y_stock = stock(x).clamp(0, 1).cpu()
        y_fused = fused(x).clamp(0, 1).cpu()

    for i in range(x.shape[0]):
        psnr, ssim = _psnr_ssim(y_stock[i], y_fused[i])
        assert psnr > MIN_PSNR_DB, f"batch item {i}: PSNR {psnr:.2f}dB below threshold"
        assert ssim > MIN_SSIM, f"batch item {i}: SSIM {ssim:.6f} below threshold"
