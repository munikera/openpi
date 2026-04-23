"""
LIBERO OpenVINO Conversion — PyTorch → ONNX → OV IR
=====================================================
Adapted from convert_droid_openvino.py for the pi05_libero model.

LIBERO model specifics (vs DROID):
  state_dim      : 8   (eef_pos 3 + axisangle 3 + gripper 2)
  action_dim     : 7   (raw LIBERO actions, padded to 32 inside model)
  action_horizon : 10  (shorter than DROID's 15)
  num_cameras    : 3   (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
  tokenizer len  : 48  (shorter prompts than DROID)
  prefix_seq_len : 596 (3 cameras × 196 SigLIP patches + 8 lang tokens, approx)

Pipeline:
  Step 1 (--export-onnx):  PyTorch  →  ONNX  (torch.onnx.export, opset 17, dynamo=False)
  Step 2 (--onnx-to-ov):   ONNX     →  OV IR FP32  (ov.convert_model)
  Step 3 (--benchmark):    Timed benchmark on OV model

Prerequisites:
  uv pip install 'onnx==1.16.1' openvino==2025.4.0

  # Convert JAX checkpoint to PyTorch first:
  python examples/convert_jax_model_to_pytorch.py \\
      --config-name pi05_libero \\
      --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \\
      --output_path   ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

Usage:
  # Full pipeline:
  python scripts/convert_libero_openvino.py --export-onnx --onnx-to-ov --benchmark

  # Step by step:
  python scripts/convert_libero_openvino.py --export-onnx
  python scripts/convert_libero_openvino.py --onnx-to-ov
  python scripts/convert_libero_openvino.py --benchmark

  # CPU-only verification (no Arc needed):
  python scripts/convert_libero_openvino.py --export-onnx --onnx-to-ov --benchmark --ov-device CPU
"""

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import tyro
import torch
import torch.nn as nn


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

    # Output directories
    onnx_dir:    str = "profiler_output/libero_onnx"    # pi05_libero.onnx
    ov_fp32_dir: str = "profiler_output/libero_fp32"    # model.xml / model.bin

    # OV inference device
    ov_device: str = "GPU"   # "GPU" = Arc, "CPU" = fallback, "AUTO" = best available

    # PyTorch device (used during export — always CPU for ONNX tracing)
    pytorch_device: str = "auto"  # auto → xpu > cuda > cpu

    # LIBERO model dimensions (must match checkpoint)
    num_cameras:     int = 3    # base + left_wrist + right_wrist (zero-padded)
    state_dim:       int = 8    # eef_pos(3) + axisangle(3) + gripper(2)
    action_horizon:  int = 10   # model config: action_horizon=10
    action_dim_full: int = 32   # internal padded action dim (model output)
    tokenizer_len:   int = 48   # max tokenizer length for libero prompts

    # Denoising steps to unroll into the graph (fixed at export time)
    num_steps: int = 10

    # Benchmark parameters
    num_warmup: int = 5
    num_iters:  int = 30

    # ONNX options
    opset_version:       int  = 17     # onnx<=1.16.1 checker supports up to 17
    do_constant_folding: bool = False  # keep False to preserve dynamic paths

    # Pipeline stages
    export_onnx: bool = False   # Step 1: PyTorch → ONNX
    onnx_to_ov:  bool = False   # Step 2: ONNX   → OV IR FP32
    benchmark:   bool = False   # Step 3: timed benchmark

    seed: int = 42


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── LIBERO inference wrapper ──────────────────────────────────────────────────

class LiberoInferenceWrapper(nn.Module):
    """Full-model ONNX-export wrapper for pi05_libero.

    Inputs (all float32 after bf16 cast, batch=1):
      images_stacked  [1, 3, 3, 224, 224]  — 3 cameras stacked on dim 1
                                              (base, left_wrist, right_wrist)
      img_masks       [1, 3]                — bool cast to float32
                                              (1,1,0 for right_wrist zero-padding)
      lang_tokens     [1, 48]               — int64
      lang_masks      [1, 48]               — bool cast to float32
      state           [1, 8]                — float32
      noise           [1, 10, 32]           — float32 (zeros for export)

    Output:
      actions         [1, 10, 7]            — float32  (first 7 of 32 action dims)

    The denoising loop is UNROLLED for `num_steps` iterations.
    KV cache stays INTERNAL to the graph.
    """

    def __init__(self, pi0_model, num_steps: int = 10):
        super().__init__()
        self.m = pi0_model
        self.num_steps = num_steps
        self.dt = -1.0 / num_steps

    def forward(
        self,
        images_stacked: torch.Tensor,   # [B, 3, 3, H, W]  float32
        img_masks: torch.Tensor,         # [B, 3]            float32 (bool)
        lang_tokens: torch.Tensor,       # [B, 48]           int64
        lang_masks: torch.Tensor,        # [B, 48]           float32 (bool)
        state: torch.Tensor,             # [B, 8]            float32
        noise: torch.Tensor,             # [B, 10, 32]       float32
    ) -> torch.Tensor:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        bsize = state.shape[0]
        device = state.device
        num_cam = images_stacked.shape[1]

        # Split stacked cameras → list
        images = [images_stacked[:, i] for i in range(num_cam)]
        img_mask_list = [img_masks[:, i].bool() for i in range(num_cam)]
        lang_masks_bool = lang_masks.bool()

        # ── Prefix forward (SigLIP + PaliGemma, fills KV cache) ────────────
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.m.embed_prefix(
            images, img_mask_list, lang_tokens, lang_masks_bool
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # Cast bool → int32 (OV GPU does not support u8 CumSum)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        prefix_att_4d = self.m._prepare_attention_masks_4d(prefix_att_2d)

        _, past_key_values = self.m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # ── Denoising loop (UNROLLED for ONNX) ─────────────────────────────
        x_t = noise
        for i in range(self.num_steps):
            time_val = 1.0 + i * self.dt
            timestep = torch.full(
                (bsize,), time_val, dtype=torch.float32, device=device
            )
            v_t = self.m.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + self.dt * v_t

        # Return only first 7 action dims (LIBERO action_dim = 7, padded to 32)
        return x_t[:, :, :7]


# ── Cast helpers ─────────────────────────────────────────────────────────────

def cast_bf16_to_fp32(model: nn.Module) -> None:
    """Cast all bf16 params/buffers to fp32 in-place (leave int32/int64 alone)."""
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            print(f"  [cast] buffer  {name}: bf16 → fp32")
            buf.data = buf.data.float()
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            print(f"  [cast] param   {name}: bf16 → fp32")
            param.data = param.data.float()


# ── Step 1: PyTorch → ONNX ───────────────────────────────────────────────────

def export_to_onnx(pi0_model, args: Args) -> Path:
    import onnx

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "pi05_libero.onnx"

    print("\n" + "=" * 60)
    print("STEP 1: PyTorch → ONNX  [pi05_libero]")
    print("=" * 60)
    print(f"  Output : {onnx_path}")
    print(f"  Steps  : {args.num_steps} (unrolled)")
    print(f"  Opset  : {args.opset_version}")

    device = "cpu"

    wrapper = LiberoInferenceWrapper(pi0_model, num_steps=args.num_steps).eval().to(device)

    print("\nCasting bf16 params/buffers → fp32 ...")
    cast_bf16_to_fp32(wrapper)
    print("Cast done.\n")

    # Dummy inputs — LIBERO shapes
    B = 1
    images_dummy  = torch.zeros(B, args.num_cameras, 3, 224, 224, dtype=torch.float32, device=device)
    # right_wrist is zero-padded → mask=0 for camera index 2
    img_masks_dummy = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    lang_tokens_dummy = torch.zeros(B, args.tokenizer_len, dtype=torch.int64, device=device)
    lang_masks_dummy  = torch.ones(B, args.tokenizer_len, dtype=torch.float32, device=device)
    state_dummy  = torch.zeros(B, args.state_dim, dtype=torch.float32, device=device)
    noise_dummy  = torch.zeros(B, args.action_horizon, args.action_dim_full, dtype=torch.float32, device=device)

    dummy_inputs = (
        images_dummy, img_masks_dummy, lang_tokens_dummy,
        lang_masks_dummy, state_dummy, noise_dummy
    )

    # Save validation inputs
    val_dir = onnx_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    torch.save({
        "images":      images_dummy.cpu(),
        "img_masks":   img_masks_dummy.cpu(),
        "lang_tokens": lang_tokens_dummy.cpu(),
        "lang_masks":  lang_masks_dummy.cpu(),
        "state":       state_dummy.cpu(),
        "noise":       noise_dummy.cpu(),
    }, val_dir / "input_tensors.pt")

    print("Running reference forward pass ...")
    with torch.no_grad():
        ref_out = wrapper(*dummy_inputs)
    torch.save(ref_out.cpu(), val_dir / "pytorch_output.pt")
    print(f"  output shape: {ref_out.shape}  dtype: {ref_out.dtype}")
    print(f"Saved inputs + reference output → {val_dir}/")

    print(f"\nExporting to ONNX (may take several minutes) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
            output_names=["actions_out"],
            opset_version=args.opset_version,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            do_constant_folding=args.do_constant_folding,
            dynamo=False,  # legacy TorchScript exporter; no onnxscript needed
            dynamic_axes={
                "images":      {0: "batch"},
                "img_masks":   {0: "batch"},
                "lang_tokens": {0: "batch"},
                "lang_masks":  {0: "batch"},
                "state":       {0: "batch"},
                "noise":       {0: "batch"},
                "actions_out": {0: "batch"},
            },
        )

    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: ✓ valid")
    except Exception as e:
        print(f"ONNX checker warning (non-fatal): {e}")
        print("  File written — proceeding.")

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\n[ONNX Export] ✓ {onnx_path}  ({size_mb:.0f} MB)")
    return onnx_path


# ── Step 2: ONNX → OV IR ─────────────────────────────────────────────────────

def onnx_to_ov_ir(onnx_path: Path, args: Args) -> Path:
    import openvino as ov

    ov_dir = Path(args.ov_fp32_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)
    xml_path = ov_dir / "model.xml"

    print("\n" + "=" * 60)
    print("STEP 2: ONNX → OV IR  [pi05_libero FP32]")
    print("=" * 60)
    print(f"  Input  : {onnx_path}")
    print(f"  Output : {xml_path}")

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, output_model=str(xml_path), compress_to_fp16=False)

    bin_mb = xml_path.with_suffix(".bin").stat().st_size / 1e6
    print(f"\n[OV Export] ✓ {xml_path}  ({bin_mb:.0f} MB bin)")
    return xml_path


# ── Step 3: Benchmark ─────────────────────────────────────────────────────────

def benchmark_ov(xml_path: Path, args: Args) -> None:
    import openvino as ov

    print("\n" + "=" * 60)
    print(f"STEP 3: Benchmark  [{args.ov_device}]  {xml_path.parent.name}/model.xml")
    print("=" * 60)

    core = ov.Core()
    print(f"  Available OV devices: {core.available_devices}")

    config = {}
    if args.ov_device.startswith("GPU"):
        config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"

    compiled  = core.compile_model(str(xml_path), device_name=args.ov_device, config=config)
    infer_req = compiled.create_infer_request()

    B = 1
    inputs_np = [
        np.zeros((B, args.num_cameras, 3, 224, 224), dtype=np.float32),  # images
        np.array([[1.0, 1.0, 0.0]], dtype=np.float32),                   # img_masks
        np.zeros((B, args.tokenizer_len), dtype=np.int64),               # lang_tokens
        np.ones ((B, args.tokenizer_len), dtype=np.float32),             # lang_masks
        np.zeros((B, args.state_dim), dtype=np.float32),                 # state
        np.zeros((B, args.action_horizon, args.action_dim_full), dtype=np.float32),  # noise
    ]

    print(f"\nWarming up ({args.num_warmup} calls) ...")
    for _ in range(args.num_warmup):
        infer_req.infer(inputs_np)
    print("Warmup done.\n")

    print(f"Running {args.num_iters} timed calls ...")
    times = []
    for _ in range(args.num_iters):
        t0 = time.time()
        infer_req.infer(inputs_np)
        times.append(time.time() - t0)

    ms = np.array(times) * 1000.0
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS  [pi05_libero]")
    print("=" * 60)
    print(f"  Device : {args.ov_device}")
    print(f"  Steps  : {args.num_steps} (unrolled)")
    print()
    print(f"  Mean   : {ms.mean():.1f} ms  ±{ms.std():.1f} ms")
    print(f"  Median : {np.percentile(ms, 50):.1f} ms")
    print(f"  P90    : {np.percentile(ms, 90):.1f} ms")
    print(f"  P95    : {np.percentile(ms, 95):.1f} ms")
    print(f"  Min    : {ms.min():.1f} ms")
    print(f"  Hz     : {1000/ms.mean():.2f}")
    print("=" * 60)
    print(f"\nBENCHMARK_CSV: openvino_{args.ov_device}_libero_fp32,"
          f"{args.config_name},steps={args.num_steps},"
          f"{ms.mean():.1f}ms,{np.percentile(ms,50):.1f}ms,"
          f"{np.percentile(ms,90):.1f}ms,{1000/ms.mean():.2f}Hz")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ov_vs_pytorch(xml_path: Path, val_dir: Path) -> float:
    import openvino as ov

    saved_inputs = torch.load(val_dir / "input_tensors.pt", weights_only=True)
    pytorch_out  = torch.load(val_dir / "pytorch_output.pt", weights_only=True).numpy()

    inputs_np = [
        saved_inputs["images"].numpy(),
        saved_inputs["img_masks"].numpy(),
        saved_inputs["lang_tokens"].numpy(),
        saved_inputs["lang_masks"].numpy(),
        saved_inputs["state"].numpy(),
        saved_inputs["noise"].numpy(),
    ]

    core = ov.Core()
    compiled  = core.compile_model(str(xml_path), device_name="CPU")
    infer_req = compiled.create_infer_request()
    infer_req.infer(inputs_np)
    ov_out = infer_req.get_output_tensor(0).data

    mse = float(np.mean((pytorch_out - ov_out) ** 2))
    print(f"\n[Validation] MSE (PyTorch vs OV): {mse:.6e}  {'✓ OK' if mse < 1e-3 else '✗ TOO HIGH'}")
    return mse


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args):
    if not any([args.export_onnx, args.onnx_to_ov, args.benchmark]):
        print("Nothing to do — pass --export-onnx, --onnx-to-ov, and/or --benchmark")
        return

    pytorch_device = _resolve_device(args.pytorch_device)
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)

    onnx_path = Path(args.onnx_dir) / "pi05_libero.onnx"
    fp32_xml  = Path(args.ov_fp32_dir) / "model.xml"
    val_dir   = Path(args.onnx_dir) / "validation"

    print("=" * 60)
    print("LIBERO OpenVINO Conversion")
    print("=" * 60)
    print(f"  Config    : {args.config_name}")
    print(f"  Ckpt      : {checkpoint_dir}")
    print(f"  Steps     : {args.num_steps}  cameras: {args.num_cameras}  state: {args.state_dim}")
    print(f"  OV device : {args.ov_device}")
    print()

    if args.export_onnx:
        print("Loading PyTorch model ...")
        t0 = time.time()
        from openpi.training import config as _config
        from openpi.policies import policy_config as _policy_config

        config = _config.get_config(args.config_name)
        policy = _policy_config.create_trained_policy(
            config, checkpoint_dir, pytorch_device=pytorch_device
        )
        pi0_model = policy._model
        pi0_model.eval()
        print(f"Loaded in {time.time()-t0:.1f}s\n")

        onnx_path = export_to_onnx(pi0_model, args)

    if args.onnx_to_ov:
        if not onnx_path.exists():
            print(f"ERROR: {onnx_path} not found. Run --export-onnx first.")
            return
        fp32_xml = onnx_to_ov_ir(onnx_path, args)

    if args.benchmark:
        if not fp32_xml.exists():
            print(f"ERROR: {fp32_xml} not found. Run --onnx-to-ov first.")
            return
        benchmark_ov(fp32_xml, args)

        if (val_dir / "pytorch_output.pt").exists():
            validate_ov_vs_pytorch(fp32_xml, val_dir)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
