"""Correctness gate for CUDA-graph-captured inference vs eager inference.

CUDA graph capture is itself a source of subtle bugs (stale memory reuse,
capturing a one-time JIT-compile or cuDNN algorithm search instead of the
steady-state kernel sequence) -- this must pass before use_cuda_graph=True
is treated as production-equivalent to the eager path, same rule as the
Triton kernel itself.
"""

from pathlib import Path

import pytest
import torch

from src.model.restorer import Restorer

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "RealESRGAN_x4plus_anime_6B.pth"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA"),
    pytest.mark.skipif(not CHECKPOINT.exists(), reason="requires the real checkpoint"),
]


def _psnr_ssim(a: torch.Tensor, b: torch.Tensor):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a_np = a.permute(1, 2, 0).numpy()
    b_np = b.permute(1, 2, 0).numpy()
    psnr = peak_signal_noise_ratio(a_np, b_np, data_range=1.0)
    ssim = structural_similarity(a_np, b_np, data_range=1.0, channel_axis=2)
    return psnr, ssim


@pytest.mark.parametrize("use_triton_kernel", [False, True])
def test_graph_matches_eager(use_triton_kernel):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)

    eager = Restorer(use_triton_kernel=use_triton_kernel, use_cuda_graph=False)
    graphed = Restorer(use_triton_kernel=use_triton_kernel, use_cuda_graph=True)

    y_eager = eager.restore_batch(x)[0]
    y_graphed = graphed.restore_batch(x)[0]

    psnr, ssim = _psnr_ssim(y_eager, y_graphed)
    assert psnr > 40.0, f"PSNR {psnr:.2f}dB below threshold"
    assert ssim > 0.995, f"SSIM {ssim:.6f} below threshold"


def test_graph_recaptures_on_shape_change():
    """restore_batch must still be correct if called with a new shape after
    a graph was already captured for a different shape (re-capture path)."""
    restorer = Restorer(use_triton_kernel=True, use_cuda_graph=True)
    eager = Restorer(use_triton_kernel=True, use_cuda_graph=False)

    torch.manual_seed(1)
    x_small = torch.rand(1, 3, 64, 64)
    x_large = torch.rand(1, 3, 128, 128)

    for x in (x_small, x_large, x_small):  # shape change, then change back
        y_graphed = restorer.restore_batch(x)[0]
        y_eager = eager.restore_batch(x)[0]
        psnr, ssim = _psnr_ssim(y_eager, y_graphed)
        assert psnr > 40.0, f"shape {tuple(x.shape)}: PSNR {psnr:.2f}dB below threshold"
        assert ssim > 0.995, f"shape {tuple(x.shape)}: SSIM {ssim:.6f} below threshold"
