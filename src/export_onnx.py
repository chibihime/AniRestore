"""Export the Real-ESRGAN anime 6B model to ONNX for the in-browser demo.

The web demo (web/index.html) runs this ONNX file client-side with ONNX
Runtime Web, so the static site needs no server. Exported from the plain
PyTorch path (use_triton_kernel=False) -- the fused Triton kernel is a CUDA
op with no ONNX equivalent. Height/width are dynamic; the page feeds it
fixed-size tiles.

Usage:
    python -m src.export_onnx --output web/model.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.model.restorer import Restorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="web/model.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = Restorer(device="cpu", use_triton_kernel=False).model
    dummy = torch.rand(1, 3, 64, 64)
    h, w = torch.export.Dim("height", min=16, max=1024), torch.export.Dim("width", min=16, max=1024)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {2: h, 3: w}},
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
    )
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Parity check against PyTorch at a different shape than the export dummy,
    # so a broken dynamic-shape export fails here rather than in the browser.
    x = torch.rand(1, 3, 96, 80)
    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": x.numpy()})[0]
    max_err = float(np.abs(actual - expected).max())
    logger.info("onnxruntime vs pytorch max abs error: %.2e", max_err)
    if actual.shape != expected.shape or max_err > 1e-3:
        raise SystemExit(f"ONNX parity check failed: shape {actual.shape} vs {expected.shape}, max err {max_err}")


if __name__ == "__main__":
    main()
