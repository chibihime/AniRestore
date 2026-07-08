"""Fused bias-add + LeakyReLU Triton kernel.

Phase 3 profiling (results/phase3_findings.md) found that PyTorch applies a
conv layer's bias as a *separate* elementwise kernel (`aten::add_`, 1:1 with
conv call count) followed by a *separate* LeakyReLU kernel (`aten::leaky_relu_`)
-- together 11.83% of total GPU time in the baseline, second only to the
convolution itself. This module fuses those two elementwise passes into one
kernel: one read of the conv output, one write of the activated result,
instead of two round trips through global memory.

The convolution itself is left alone (still cuDNN's tuned Winograd kernel,
65.9% of GPU time and not a sane target to reimplement in Triton) -- callers
must invoke conv2d with bias=None and pass the bias tensor here separately.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    # Triton requires CUDA and isn't installable on e.g. local Apple Silicon
    # dev machines. The CPU fallback path below doesn't need it at all, so
    # only fail if something actually tries to run the GPU kernel.
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_bias_leaky_relu_kernel(
        x_ptr, bias_ptr, out_ptr,
        numel, hw, num_channels,
        negative_slope,
        ACTIVATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # NCHW contiguous layout: flat_idx = ((n*C + c)*H + h)*W + w, so
        # flat_idx // (H*W) == n*C + c regardless of n, and % C recovers c.
        channel = (offs // hw) % num_channels
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)

        y = x + bias
        if ACTIVATE:
            y = tl.where(y >= 0, y, y * negative_slope)

        tl.store(out_ptr + offs, y, mask=mask)


def fused_bias_leaky_relu(
    x: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.2,
    activate: bool = True,
) -> torch.Tensor:
    """y = leaky_relu(x + bias, negative_slope) if activate else x + bias.

    x: NCHW conv output *without* bias applied (pass bias=None to conv2d).
    bias: (C,) tensor, one value per channel, broadcast over N/H/W.

    Falls back to plain PyTorch ops on CPU (Triton requires CUDA) so the
    model still runs for local dev/testing; the actual fused kernel only
    executes on GPU, which is also where the profiled bottleneck exists.
    """
    if not x.is_cuda:
        y = x + bias.view(1, -1, 1, 1)
        return F.leaky_relu(y, negative_slope) if activate else y

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed -- required to run the fused kernel on CUDA tensors")

    assert x.ndim == 4, "expected NCHW input"
    x = x.contiguous()
    bias = bias.contiguous()
    n, c, h, w = x.shape
    assert bias.shape == (c,), f"bias shape {tuple(bias.shape)} != (C,)={c}"

    out = torch.empty_like(x)
    numel = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    _fused_bias_leaky_relu_kernel[grid](
        x, bias, out,
        numel, h * w, c,
        negative_slope,
        ACTIVATE=activate,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
