"""Verify the environment is ready for AniRestore GPU work.

Run this first on any new machine/notebook (e.g. Kaggle) before touching
model or kernel code. Exits non-zero if any required check fails.
"""

import sys


def check_torch_cuda():
    import torch

    print(f"torch version: {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        return False

    print(f"CUDA version (torch build): {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    return True


def check_triton():
    import torch
    import triton
    import triton.language as tl

    print(f"triton version: {triton.__version__}")

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    _add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()

    expected = x + y
    max_err = (out - expected).abs().max().item()
    print(f"Triton trivial kernel max error vs torch: {max_err:.2e}")
    return max_err < 1e-5


def main():
    results = {}

    try:
        results["torch+cuda"] = check_torch_cuda()
    except Exception as e:
        print(f"[FAIL] torch/cuda check raised: {e}")
        results["torch+cuda"] = False

    if results["torch+cuda"]:
        try:
            results["triton"] = check_triton()
        except Exception as e:
            print(f"[FAIL] triton check raised: {e}")
            results["triton"] = False
    else:
        print("Skipping Triton check — no CUDA GPU available.")
        results["triton"] = False

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")

    if not all(results.values()):
        sys.exit(1)
    print("\nAll checks passed — environment is ready.")


if __name__ == "__main__":
    main()
