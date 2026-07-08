"""Unit tests for the fused bias+LeakyReLU Triton kernel.

Compares the Triton kernel's output directly against a plain-PyTorch
reference (conv bias-add + LeakyReLU done as two separate ops) across a
matrix of shapes, batch sizes, and both activate=True/False, since Triton
requires CUDA and won't run on the local (CPU-only) dev machine.
"""

import pytest
import torch
import torch.nn.functional as F

from src.kernels.fused_bias_act import fused_bias_leaky_relu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")


def _reference(x: torch.Tensor, bias: torch.Tensor, activate: bool) -> torch.Tensor:
    y = x + bias.view(1, -1, 1, 1)
    return F.leaky_relu(y, 0.2) if activate else y


@pytest.mark.parametrize("n,c,h,w", [
    (1, 3, 16, 16),
    (2, 32, 64, 64),
    (4, 64, 32, 32),
    (1, 64, 17, 33),   # odd, non-power-of-2 spatial dims
    (3, 96, 256, 256),  # matches RDB's widest concatenated channel count
])
@pytest.mark.parametrize("activate", [True, False])
def test_matches_reference(n, c, h, w, activate):
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, device="cuda")
    bias = torch.randn(c, device="cuda")

    got = fused_bias_leaky_relu(x, bias, negative_slope=0.2, activate=activate)
    expected = _reference(x, bias, activate)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_negative_slope_is_respected():
    x = torch.tensor([[[[-2.0, 2.0]]]], device="cuda")
    bias = torch.tensor([0.0], device="cuda")
    out = fused_bias_leaky_relu(x, bias, negative_slope=0.1, activate=True)
    torch.testing.assert_close(out, torch.tensor([[[[-0.2, 2.0]]]], device="cuda"))


def test_non_contiguous_input_still_correct():
    torch.manual_seed(1)
    x_full = torch.randn(2, 8, 20, 20, device="cuda")
    x = x_full[:, :, ::2, ::2]  # non-contiguous slice
    bias = torch.randn(8, device="cuda")

    got = fused_bias_leaky_relu(x, bias, activate=True)
    expected = _reference(x, bias, activate=True)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
