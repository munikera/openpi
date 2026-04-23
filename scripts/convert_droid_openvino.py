"""
DROID OpenVINO Conversion — Intel's Three-Step Approach
========================================================
Adapted from:
  https://github.com/open-edge-platform/edge-ai-suites/tree/main/
  robotics-ai-suite/pipelines/vla-pi0.5-openvino

Pipeline:
  Step 1 (--export-onnx):   PyTorch  →  ONNX         (torch.onnx.export, opset 20)
  Step 2 (--onnx-to-ov):    ONNX     →  OV IR FP32    (ov.convert_model)
  Step 3 (--benchmark):     Run timed inference on OV model

Key differences from benchmark_droid_openvino.py (Path 2 direct ov.convert_model):
  - Uses ONNX as a stable intermediate representation (avoids TorchScript tracing issues)
  - Exports the FULL model in one graph (prefix + N denoising steps unrolled)
  - Casts all bf16 params/buffers → fp32 before export (same as Intel's script)
  - No Python KV-cache boundary — KV cache stays internal to the ONNX graph

Prerequisites:
  pip install onnx==1.16.1 openvino==2025.4.0
  # Note: onnx>=1.17 requires ml_dtypes>=0.5.0 (float4_e2m1fn). Use 1.16.1 to
  # avoid conflicts with the ml_dtypes version pinned by the openpi venv.

  # Convert JAX checkpoint to PyTorch first:
  python examples/convert_jax_model_to_pytorch.py \\
      --config-name pi05_droid \\
      --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \\
      --output_path   ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

Usage:
  # Full pipeline in one command:
  python scripts/convert_droid_openvino.py --export-onnx --onnx-to-ov --benchmark

  # Step by step:
  python scripts/convert_droid_openvino.py --export-onnx
  python scripts/convert_droid_openvino.py --onnx-to-ov
  python scripts/convert_droid_openvino.py --benchmark

  # CPU-only verification (no Arc needed):
  python scripts/convert_droid_openvino.py --export-onnx --onnx-to-ov --benchmark --ov-device CPU

Known issues / mitigations — see TROUBLESHOOTING at the bottom.
"""

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import tyro
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    config_name: str = "pi05_droid"
    checkpoint_dir: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch"

    # Output directories (mirrors Intel's layout)
    onnx_dir:     str = "profiler_output/openvino_onnx"    # pi05.onnx
    ov_fp32_dir:  str = "profiler_output/openvino_fp32"    # model.xml / model.bin

    # Which OpenVINO device to run on for benchmarking.
    # "GPU"  = Intel Arc (or any OV GPU plugin device)
    # "CPU"  = OV CPU plugin (always available, useful for verification)
    # "AUTO" = OV auto-selects best available device
    ov_device: str = "GPU"

    # PyTorch device used during export (always CPU for ONNX tracing)
    # Also used for prefix forward during --benchmark
    pytorch_device: str = "auto"   # auto → xpu > cuda > cpu

    # Number of denoising steps to unroll into the ONNX graph (fixed at export time)
    num_steps: int = 10

    # Benchmark parameters
    num_warmup: int = 5
    num_iters: int = 30

    # ONNX export options
    opset_version: int = 17  # 17 is widely supported; onnx<=1.16.1 checker fails on opset>=18
    do_constant_folding: bool = False   # Intel uses False; True may fold away dynamic paths

    # Pipeline stages (at least one must be set)
    export_onnx:   bool = False   # Step 1: PyTorch → ONNX
    onnx_to_ov:    bool = False   # Step 2: ONNX → OV IR FP32
    benchmark:     bool = False   # Step 3: benchmark the OV model

    seed: int = 42


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Inference wrapper ─────────────────────────────────────────────────────────

class Pi0DroidInferenceWrapper(nn.Module):
    """Full-model ONNX-export wrapper for pi05_droid.

    Inputs  (all float32 after bf16 cast, batch=1):
      images_stacked  [1, num_cameras, 3, 224, 224]  — cameras stacked on dim 1
      img_masks       [1, num_cameras]                — bool cast to float32
      lang_tokens     [1, 200]                        — int64
      lang_masks      [1, 200]                        — bool cast to float32
      state           [1, 8]                          — robot state
      noise           [1, 15, 32]                     — pre-sampled noise (zeros for export)

    Output:
      actions         [1, 15, 32]                     — predicted actions

    The denoising loop is UNROLLED for `num_steps` iterations so that ONNX
    sees a static graph (no Python while-loop control flow).
    KV cache stays INTERNAL to the graph — no split at cache boundary.
    """

    def __init__(self, pi0_model, num_steps: int = 10):
        super().__init__()
        self.m = pi0_model
        self.num_steps = num_steps
        self.dt = -1.0 / num_steps

    def forward(
        self,
        images_stacked: torch.Tensor,   # [B, num_cam, 3, H, W] float32
        img_masks: torch.Tensor,         # [B, num_cam]          float32 (bool)
        lang_tokens: torch.Tensor,       # [B, 200]              int64
        lang_masks: torch.Tensor,        # [B, 200]              float32 (bool)
        state: torch.Tensor,             # [B, 8]                float32
        noise: torch.Tensor,             # [B, 15, 32]           float32
    ) -> torch.Tensor:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        bsize = state.shape[0]
        device = state.device
        num_cam = images_stacked.shape[1]

        # Split stacked cameras → list (as embed_prefix expects)
        images = [images_stacked[:, i] for i in range(num_cam)]
        img_mask_list = [img_masks[:, i].bool() for i in range(num_cam)]
        lang_masks_bool = lang_masks.bool()

        # ── Prefix forward (SigLIP + PaliGemma, fills KV cache) ────────────
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.m.embed_prefix(
            images, img_mask_list, lang_tokens, lang_masks_bool
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # Cast bool → int32 before cumsum: OV GPU doesn't support u8 CumSum
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

        return x_t


# ── Helpers ───────────────────────────────────────────────────────────────────

def cast_bf16_to_fp32(model: nn.Module) -> None:
    """Cast all bf16 parameters and buffers to fp32 in-place.

    Intel's script does this before ONNX export so that the graph uses only
    dtypes that ONNX opset 20 handles cleanly.  int32/int64 are left alone
    (needed for Gather, indexing, etc.).
    """
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            print(f"  [cast] buffer  {name}: bf16 → fp32")
            buf.data = buf.data.float()
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            print(f"  [cast] param   {name}: bf16 → fp32")
            param.data = param.data.float()


def make_synthetic_obs(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":      rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position":        rng.random(7).astype(np.float64),
        "observation/gripper_position":      rng.random(1).astype(np.float64),
        "prompt":                            "pick up the cup",
    }


def _preprocess_to_tensors(pi0_model, obs: dict, device: str):
    """Run preprocess + build stacked tensors ready for Pi0DroidInferenceWrapper."""
    from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
    from openpi.policies import droid_policy as _dp

    policy_input = _dp.DroidInputs.from_dict(obs)
    processed = _preprocessing.preprocess_observation_pytorch(policy_input, train=False)

    images_list = list(processed.images.values())       # list of [1,3,H,W]
    img_masks_list = list(processed.image_masks.values())  # list of [1]
    lang_tokens = processed.tokenized_prompt             # [1, 200]
    lang_masks  = processed.tokenized_prompt_mask        # [1, 200]
    state       = processed.state                        # [1, 8]

    num_cam = len(images_list)
    images_stacked = torch.stack(images_list, dim=1)          # [1, num_cam, 3, H, W]
    img_masks_stacked = torch.stack(img_masks_list, dim=1)    # [1, num_cam]

    images_stacked   = images_stacked.float().to(device)
    img_masks_stacked = img_masks_stacked.float().to(device)
    lang_tokens      = lang_tokens.to(device)
    lang_masks       = lang_masks.float().to(device)
    state            = state.float().to(device)

    return images_stacked, img_masks_stacked, lang_tokens, lang_masks, state


# ── Step 1: Export to ONNX ────────────────────────────────────────────────────

def export_to_onnx(pi0_model, args: Args):
    """Step 1 — PyTorch → ONNX (follows Intel's convert_pytorch_onnx.py)."""
    import onnx

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "pi05_droid.onnx"

    print("\n" + "=" * 60)
    print("STEP 1: PyTorch → ONNX")
    print("=" * 60)
    print(f"  Output : {onnx_path}")
    print(f"  Steps  : {args.num_steps} (unrolled in graph)")
    print(f"  Opset  : {args.opset_version}")
    print(f"  do_constant_folding: {args.do_constant_folding}")

    device = "cpu"   # OV traces on CPU; move model there

    # ── Build wrapper + cast bf16 → fp32 ─────────────────────────────────
    wrapper = Pi0DroidInferenceWrapper(pi0_model, num_steps=args.num_steps).eval()
    wrapper = wrapper.to(device)

    print("\nCasting bf16 parameters and buffers to fp32 ...")
    cast_bf16_to_fp32(wrapper)
    print("Cast done.\n")

    # ── Build dummy inputs (all fp32, no randomness) ──────────────────────
    # Shapes match DROID pi0.5 (batch=1):
    #   num_cameras = 2 (exterior_left + wrist_left)
    #   lang tokens = 200 (tokenizer_max_length)
    #   state dim   = 8
    #   action_horizon = 15, action_dim = 32
    B        = 1
    num_cam  = 2
    T_lang   = 200
    H, W     = 224, 224

    images_dummy   = torch.zeros(B, num_cam, 3, H, W, dtype=torch.float32, device=device)
    img_masks_dummy = torch.ones(B, num_cam, dtype=torch.float32, device=device)
    lang_tokens_dummy = torch.zeros(B, T_lang, dtype=torch.int64, device=device)
    lang_masks_dummy  = torch.ones(B, T_lang, dtype=torch.float32, device=device)
    state_dummy  = torch.zeros(B, 8, dtype=torch.float32, device=device)
    noise_dummy  = torch.zeros(B, 15, 32, dtype=torch.float32, device=device)

    dummy_inputs = (
        images_dummy, img_masks_dummy, lang_tokens_dummy,
        lang_masks_dummy, state_dummy, noise_dummy
    )

    # Save inputs for validation (mirrors Intel's approach)
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
    print(f"Saved dummy inputs → {val_dir}/input_tensors.pt")

    # Run once to get reference output for validation
    print("Running reference forward pass ...")
    with torch.no_grad():
        ref_out = wrapper(*dummy_inputs)
    torch.save(ref_out.cpu(), val_dir / "pytorch_output.pt")
    print(f"Saved reference output → {val_dir}/pytorch_output.pt")
    print(f"  output shape: {ref_out.shape}  dtype: {ref_out.dtype}")

    # ── ONNX export ───────────────────────────────────────────────────────
    print(f"\nExporting to ONNX (this may take several minutes) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            input_names=[
                "images", "img_masks", "lang_tokens",
                "lang_masks", "state", "noise"
            ],
            output_names=["actions_out"],
            opset_version=args.opset_version,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            do_constant_folding=args.do_constant_folding,
            # Force the legacy TorchScript-based exporter (torch >= 2.1 defaults to
            # the dynamo exporter which requires the optional 'onnxscript' package).
            dynamo=False,
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

    # Verify the ONNX model is well-formed (non-fatal: onnx<=1.16.1 may reject opset>=18)
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: ✓ model is valid")
    except Exception as e:
        print(f"ONNX checker warning (non-fatal): {e}")
        print("  The .onnx file was written — proceeding to OV conversion.")
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\n[ONNX Export] ✓ {onnx_path}  ({size_mb:.0f} MB)")
    return onnx_path


# ── Step 2: ONNX → OV IR ─────────────────────────────────────────────────────

def onnx_to_ov_ir(onnx_path: Path, args: Args, compress_to_fp16: bool = False):
    """Step 2 — ONNX → OV IR FP32 (follows Intel's onnx_to_ov_ir.py)."""
    import openvino as ov

    ov_dir = Path(args.ov_fp32_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)
    xml_path = ov_dir / "model.xml"

    print("\n" + "=" * 60)
    print("STEP 2: ONNX → OV IR")
    print("=" * 60)
    print(f"  Input  : {onnx_path}")
    print(f"  Output : {xml_path}")
    print(f"  FP16 compress: {compress_to_fp16}")

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(
        ov_model,
        output_model=str(xml_path),
        compress_to_fp16=compress_to_fp16,
    )

    size_mb = xml_path.stat().st_size / 1e6
    bin_mb  = xml_path.with_suffix(".bin").stat().st_size / 1e6
    print(f"\n[OV Export] ✓ {xml_path}  ({size_mb:.1f} MB xml + {bin_mb:.0f} MB bin)")
    return xml_path


# ── Step 3: Benchmark ─────────────────────────────────────────────────────────

def benchmark_ov(xml_path: Path, args: Args, rng: np.random.Generator):
    """Step 4 — timed benchmark of the compiled OV model."""
    import openvino as ov

    print("\n" + "=" * 60)
    print(f"STEP 4: Benchmark  [{args.ov_device}]  {xml_path.parent.name}/model.xml")
    print("=" * 60)

    core = ov.Core()
    print(f"  Available OV devices: {core.available_devices}")

    config = {}
    if args.ov_device.startswith("GPU"):
        config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"

    compiled   = core.compile_model(str(xml_path), device_name=args.ov_device, config=config)
    infer_req  = compiled.create_infer_request()

    # Build dummy numpy inputs
    B, num_cam, T_lang = 1, 2, 200
    inputs_np = [
        np.zeros((B, num_cam, 3, 224, 224), dtype=np.float32),  # images
        np.ones ((B, num_cam),              dtype=np.float32),  # img_masks
        np.zeros((B, T_lang),               dtype=np.int64),    # lang_tokens
        np.ones ((B, T_lang),               dtype=np.float32),  # lang_masks
        np.zeros((B, 8),                    dtype=np.float32),  # state
        np.zeros((B, 15, 32),               dtype=np.float32),  # noise
    ]

    # Warmup
    print(f"\nWarming up ({args.num_warmup} calls) ...")
    for _ in range(args.num_warmup):
        infer_req.infer(inputs_np)
    print("Warmup done.\n")

    # Timed runs
    print(f"Running {args.num_iters} timed calls ...")
    times = []
    for _ in range(args.num_iters):
        t0 = time.time()
        infer_req.infer(inputs_np)
        times.append(time.time() - t0)

    ms = np.array(times) * 1000.0
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Model  : {xml_path}")
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
    print(f"\nBENCHMARK_CSV: openvino_{args.ov_device}_{xml_path.parent.name},"
          f"{args.config_name},steps={args.num_steps},"
          f"{ms.mean():.1f}ms,{np.percentile(ms,50):.1f}ms,"
          f"{np.percentile(ms,90):.1f}ms,{1000/ms.mean():.2f}Hz")


# ── Optional validation ───────────────────────────────────────────────────────

def validate_ov_vs_pytorch(xml_path: Path, val_dir: Path, args: Args):
    """Compare OV model output against saved PyTorch reference (MSE should be < 1e-3)."""
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
        print("Nothing to do — pass at least one of: --export-onnx --onnx-to-ov --benchmark")
        return

    pytorch_device = _resolve_device(args.pytorch_device)
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    rng = np.random.default_rng(args.seed)

    onnx_path  = Path(args.onnx_dir)  / "pi05_droid.onnx"
    fp32_xml   = Path(args.ov_fp32_dir) / "model.xml"
    val_dir    = Path(args.onnx_dir) / "validation"

    print("=" * 60)
    print("DROID OpenVINO Conversion (Intel Three-Step)")
    print("=" * 60)
    print(f"  Config    : {args.config_name}")
    print(f"  Ckpt      : {checkpoint_dir}")
    print(f"  Steps     : {args.num_steps}")
    print(f"  OV device : {args.ov_device}  (benchmark)")
    print()

    # Load PyTorch model only when needed
    pi0_model = None
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

    # Step 1: PyTorch → ONNX
    if args.export_onnx:
        onnx_path = export_to_onnx(pi0_model, args)

    # Step 2: ONNX → OV IR FP32
    if args.onnx_to_ov:
        if not onnx_path.exists():
            print(f"ERROR: {onnx_path} not found. Run --export-onnx first.")
            return
        fp32_xml = onnx_to_ov_ir(onnx_path, args, compress_to_fp16=False)

    # Step 3: Benchmark
    if args.benchmark:
        if not fp32_xml.exists():
            print(f"ERROR: {fp32_xml} not found. Run --onnx-to-ov first.")
            return
        benchmark_ov(fp32_xml, args, rng)

        # Optional validation (if reference outputs exist)
        if (val_dir / "pytorch_output.pt").exists():
            validate_ov_vs_pytorch(fp32_xml, val_dir, args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)


# ─── TROUBLESHOOTING ──────────────────────────────────────────────────────────
#
# 1. "torch.onnx.export fails with graph break / unsupported op"
#    The most common culprits in pi05_droid:
#      a. _apply_checkpoint (gradient checkpointing) — disable with:
#           pi0_model.config.gradient_checkpointing = False
#      b. DynamicCache list operations — they are inlined by trace, usually fine
#      c. make_att_2d_masks / _prepare_attention_masks_4d — uses torch.triu / bool masks,
#         generally traceable.  If not, replace with hardcoded static mask.
#      d. create_sinusoidal_pos_embedding — uses torch.arange, fine for tracing.
#    Try adding verbose=True to torch.onnx.export to see which op fails.
#
# 2. "ONNX checker error: unrecognised op"
#    Reduce opset_version to 17 or 18 (ONNX_ATEN_FALLBACK covers missing ops).
#    Or use operator_export_type=ONNX (stricter, may need more patches).
#
# 3. "OV convert_model fails on ATen ops"
#    OpenVINO 2025.4+ can handle most ATen fallback ops.  If an op is
#    unsupported, patch it in the wrapper (replace with an OV-supported equivalent).
#
# 4. "NNCF ImportError / version mismatch"
#    nncf==2.19.0 requires openvino==2025.4.0 exactly.
#    Pin:  pip install openvino==2025.4.0 nncf==2.19.0 onnx==1.20.0
#
# 5. "OV GPU plugin not found"
#    pip install openvino[gpu]   or install Intel OpenVINO toolkit:
#    https://docs.openvino.ai/latest/get_started.html
#
# 6. "Numerical mismatch (MSE > 1e-3)"
#    OV GPU may use fp16 internally.  Force fp32 precision:
#      config = {"INFERENCE_PRECISION_HINT": "f32"}
#      compiled = core.compile_model(xml_path, device_name=ov_device, config=config)
#    Or use the FP32 OV model (--ov-model fp32) instead of INT8.
#
# 7. "Export takes too long / OOM on small machines"
#    The unrolled 10-step loop makes the graph ~10× larger.  Reduce with:
#      --num-steps 1   (export single-step model, run loop in Python)
#    This trades graph compilation efficiency for export time.
