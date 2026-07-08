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
    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        use_triton_kernel: bool = False,
        use_cuda_graph: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            # Fixed input shape per benchmark/demo call site -- let cuDNN
            # autotune and pick the fastest conv algorithm for that shape
            # instead of its default heuristic choice. One-time re-tuning
            # cost per distinct shape, amortized over repeated calls.
            torch.backends.cudnn.benchmark = True
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            use_triton_kernel=use_triton_kernel,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

        self.use_cuda_graph = use_cuda_graph
        if use_cuda_graph and self.device.type != "cuda":
            raise ValueError("use_cuda_graph requires a CUDA device")
        self._graph = None
        self._graph_shape = None
        self._static_input = None
        self._static_output = None

    @torch.no_grad()
    def restore(self, image: torch.Tensor) -> torch.Tensor:
        """image: CHW float32 tensor in [0, 1]. Returns CHW float32 in [0, 1], 4x upscaled."""
        return self.restore_batch(image.unsqueeze(0))[0]

    @torch.no_grad()
    def restore_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """batch: NCHW float32 tensor in [0, 1]. Returns NCHW float32 in [0, 1], 4x upscaled."""
        batch = batch.to(self.device)
        if self.use_cuda_graph:
            out = self._replay(batch)
        else:
            out = self.model(batch).clamp(0, 1)
        return out.cpu()

    def _replay(self, batch: torch.Tensor) -> torch.Tensor:
        shape = tuple(batch.shape)
        if self._graph is None or self._graph_shape != shape:
            self._capture(shape)
        self._static_input.copy_(batch)
        self._graph.replay()
        return self._static_output.clone()

    def _capture(self, shape: tuple) -> None:
        """Capture the forward pass as a CUDA graph for a fixed input shape.

        CUDA graphs eliminate per-op Python/driver launch overhead by
        recording the exact kernel sequence once and replaying it -- see
        results/phase4_findings.md for why this matters: fusing bias+
        LeakyReLU cut GPU compute time for that op by 53%, but at batch=1
        that saving was fully absorbed by per-launch dispatch overhead
        (~1800 small kernel launches/frame). Graph replay avoids
        re-dispatching each of those launches from Python every call.

        Must warm up on a *side stream* before capture (PyTorch's documented
        pattern) so cuDNN algorithm selection and Triton's first-call JIT
        compilation both happen outside the captured region -- capturing a
        JIT compile or algorithm search would bake in the wrong thing.
        """
        static_input = torch.zeros(shape, device=self.device)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.model(static_input).clamp(0, 1)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(static_input).clamp(0, 1)

        self._graph = graph
        self._graph_shape = shape
        self._static_input = static_input
        self._static_output = static_output
