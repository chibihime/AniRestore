"""Gradio demo: anime cover-art restoration via the fused Triton kernel.

Run locally:
    python app.py
Then open the printed local URL (usually http://127.0.0.1:7860).

Uses the shipped Phase 4 configuration (results/phase4_findings.md):
the fused Triton kernel without CUDA graph capture -- the only configuration
with a clean, statistically significant latency win on this project's
hardware. On CPU (no CUDA available) this transparently falls back to plain
PyTorch ops (src/kernels/fused_bias_act.py), so the demo still runs correctly,
just without the speedup.
"""

import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.data_loader import pil_to_tensor, tensor_to_pil
from src.model.restorer import Restorer

MAX_INPUT_SIDE = 512  # cap input resolution so the 4x output stays a reasonable size
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

restorer = Restorer(use_triton_kernel=True, use_cuda_graph=False)


def _cap_size(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def restore(image: Image.Image):
    if image is None:
        return None, ""

    image = _cap_size(image, MAX_INPUT_SIDE)
    x = pil_to_tensor(image)

    if restorer.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = restorer.restore(x)
    if restorer.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_pil = tensor_to_pil(out)
    caption = (
        f"**{elapsed_ms:.0f} ms** on `{restorer.device}` (fused Triton kernel) "
        f"— {image.width}x{image.height} → {out_pil.width}x{out_pil.height}"
    )
    return out_pil, caption


with gr.Blocks(title="AniRestore") as demo:
    gr.Markdown(
        "# AniRestore\n"
        "4x anime cover-art restoration with Real-ESRGAN, accelerated by a custom "
        "fused Triton kernel (see [phase4_findings.md](results/phase4_findings.md) "
        "for the benchmark). Upload your own image or pick a sample below, then "
        "click Restore."
    )
    with gr.Row():
        input_image = gr.Image(type="pil", label="Original")
        output_image = gr.Image(type="pil", label="Restored (4x)")

    restore_btn = gr.Button("Restore", variant="primary")
    inference_time = gr.Markdown()

    gr.Examples(
        examples=[str(p) for p in sorted(EXAMPLES_DIR.iterdir())] if EXAMPLES_DIR.exists() else [],
        inputs=input_image,
        label="Sample cover art",
    )

    restore_btn.click(fn=restore, inputs=input_image, outputs=[output_image, inference_time])


if __name__ == "__main__":
    demo.launch()
