// Tiled 4x upscaling with the exported ONNX model (src/export_onnx.py).
//
// Runs the image through the model in fixed-size, overlapping tiles rather
// than all at once: the upsampling layers hold 64-channel feature maps at
// the *output* resolution, which for a full 512x512 input is well over a
// gigabyte -- more than a browser tab can reliably allocate. Each tile is
// read with `pad` pixels of context on every side and only its centre is
// kept, so tile seams don't show (pad=32 stays within 4/255 of a single
// whole-image pass; pad=12 left visible seams). Windows are shifted inward at the image
// edges instead of shrunk, so every run sees the same input shape (WebGPU
// recompiles per distinct shape).

export async function upscale(session, ort, rgba, width, height, opts = {}) {
  const { tile = 128, pad = 32, scale = 4, onProgress = () => {} } = opts;
  const outW = width * scale;
  const outH = height * scale;
  const out = new Uint8ClampedArray(outW * outH * 4);
  const winW = Math.min(width, tile + 2 * pad);
  const winH = Math.min(height, tile + 2 * pad);
  const cols = Math.ceil(width / tile);
  const rows = Math.ceil(height / tile);
  const input = new Float32Array(3 * winW * winH);
  const plane = winW * winH;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Core region this tile is responsible for, and the window read around it.
      const cx0 = c * tile, cy0 = r * tile;
      const cx1 = Math.min(cx0 + tile, width), cy1 = Math.min(cy0 + tile, height);
      const wx0 = Math.max(0, Math.min(cx0 - pad, width - winW));
      const wy0 = Math.max(0, Math.min(cy0 - pad, height - winH));

      for (let y = 0; y < winH; y++) {
        for (let x = 0; x < winW; x++) {
          const src = ((wy0 + y) * width + (wx0 + x)) * 4;
          const dst = y * winW + x;
          input[dst] = rgba[src] / 255;
          input[plane + dst] = rgba[src + 1] / 255;
          input[2 * plane + dst] = rgba[src + 2] / 255;
        }
      }

      const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, winH, winW]) };
      const result = await session.run(feeds);
      const data = result[session.outputNames[0]].data;
      const tw = winW * scale, tplane = tw * winH * scale;

      for (let y = cy0 * scale; y < cy1 * scale; y++) {
        const ty = y - wy0 * scale;
        for (let x = cx0 * scale; x < cx1 * scale; x++) {
          const t = ty * tw + (x - wx0 * scale);
          const o = (y * outW + x) * 4;
          out[o] = data[t] * 255;
          out[o + 1] = data[tplane + t] * 255;
          out[o + 2] = data[2 * tplane + t] * 255;
          out[o + 3] = 255;
        }
      }
      onProgress((r * cols + c + 1) / (rows * cols));
    }
  }
  return { data: out, width: outW, height: outH };
}
