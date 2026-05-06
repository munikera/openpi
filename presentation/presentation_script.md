# Enable VLA Models on Intel Arc Pro B70
## Presentation Structure & Speaker Script

---

## Slide 1 — Title

**Title:** Enable VLA Models on Intel Arc Pro  
**Subtitle:** π0.5 · OpenVLA · LIBERO & DROID Benchmarks · OpenVINO Optimization  
**Footer:** Arc Pro B70 · PyTorch XPU → OpenVINO · Real-Time MuJoCo Inference

**Script:**
> "Today I want to walk you through a project to enable Vision-Language-Action models — the class of AI that directly controls robots — on Intel's Arc Pro B70. We'll cover what VLA models are, how we got both OpenVLA and π0.5 running on XPU, how we profiled and benchmarked them against an NVIDIA RTX PRO 4000, and how OpenVINO helped us hit real-time performance."

---

## Slide 2 — Agenda

1. Strategic Context
2. Why VLAs? — The Generalization Problem
3. From LLM to VLA
4. Models on Arc Pro B70
5. Enabling on Arc Pro B70
6. LIBERO Benchmark
7. Benchmark Results
8. PyTorch Profiling
9. OpenVINO Optimization
10. Breaking 100 ms
11. Live Demo
12. Q&A

---

## Slide 3 — Strategic Context

**Key points:**
- Request from Mark Iskra (OpenAI AE) — enable VLA models on B70 as NVIDIA workstation alternative
- ECG team already has π0.5 on Lunar Lake at ~10 fps using 8 NPUs; working with Physical Intelligence
- Our goal: validate Arc Pro B70 as a VLA development workstation — run under 100 ms, compare vs RTX PRO 4000

**Script:**
> "This started with a meeting with an OpenAI account executive who asked whether the B70 could serve as a development workstation for Physical AI. Separately, Intel's ECG team was already running π0.5 on embedded NPU hardware and wanted to align. Our job was to close the loop for the workstation tier."

---

## Slide 4 — Why VLAs? — The Generalization Problem

### The Problem: Traditional Robot Policies Can't Generalize

Learned policies trained on individual skills can extrapolate to new initial conditions (object position, lighting) but **fail to generalize** to:
- Scene distractors or novel objects
- Unseen task instructions
- Robots or environments not seen during training

### Root Cause: The Data Gap

Robot manipulation datasets are tiny compared to internet-scale vision/language data:

| Source | Scale |
|---|---|
| Largest robot datasets (e.g., Open X-Embodiment) | 100K – 1M examples |
| Vision-language pretraining data (CLIP, SigLIP, Llama) | Billions of examples |

This imbalance suggests an opportunity: **use foundation models as a core building block** for robot policies, rather than training from scratch.

### Why Not Just Use Closed VLAs?

Even early capable VLAs (RT-2, etc.) had two blockers for widespread adoption:
1. **Closed weights** — no visibility into architecture, training data, or procedures
2. **No deployment guidance** — especially for new robots, environments, or commodity hardware (consumer GPUs)

→ The field needs **open-source, generalist VLAs** that support effective fine-tuning.

### How SigLIP + LLM → VLA

A VLA is simply a Vision-Language Model (VLM) fine-tuned to output robot actions:

```
Camera Images  →  [SigLIP Visual Encoder]  →  Visual tokens (patch embeddings)
                                                        ↓
Task Instruction →  [LLM Backbone: Llama / Gemma]  →  Contextual understanding
                                                        ↓
                          [Action Head]  →  Robot joint commands
```

- **SigLIP** brings rich visual priors from internet-scale image-text training — it already "knows" what objects look like
- **The LLM** brings semantic reasoning — it understands language instructions and can relate them to visual context
- **The action head** is trained on robot data; the rest benefits from foundation model priors
- Only a small fraction of the model learns from robot trajectories → generalization is inherited from the foundation model

**Script:**
> "So why VLAs? Traditional learned robot policies — imitation learning, RL — are brittle. They can handle variations they saw during training, but drop something unexpected on the table and they fail. The root cause is simple: robot datasets top out at a million examples, while the internet has billions of labeled images and text. VLAs solve this by taking a model like SigLIP + Llama — already trained on that internet-scale data — and fine-tuning just the action head on robot demonstrations. SigLIP understands what objects look like. The LLM understands what 'pick up the red block' means. You connect them to a small action head, fine-tune on robot data, and now your robot policy inherits all those visual and semantic priors — which is exactly what enables generalization to novel objects and tasks."

---

## Slide 5 — From LLM to VLA

**Key points:**
- LLM: text tokens in → text tokens out, autoregressive
- VLM: adds a visual encoder (SigLIP/ViT) — image patches become tokens
- VLA: adds an action head — model sees the world and outputs motor commands directly

**Script:**
> "VLA models are a natural evolution. You take a language model, add a vision encoder so it can understand camera images, then add an action head so it can output robot joint commands rather than text. The key insight is that the model learns to act directly from examples — no hand-coded planners, no reward shaping."

---

## Slide 6 — Models Enabled on Arc Pro B70

**Both models running on XPU:**

### OpenVLA (7B params)
- **Architecture:** Prismatic VLM — SigLIP visual encoder + Llama-2 7B language backbone
- **Action head:** Discrete token bins (256 bins per joint dimension), autoregressive decoding
- **Training data:** Open-X-Embodiment (diverse robot trajectories)
- One forward pass → action tokens → decoded to joint angles

### π0.5 (3B params)
- **Architecture:** PaliGemma — SigLIP + Gemma 2B language model
- **Action head:** Flow-matching expert — iteratively denoises random noise → smooth action trajectory
- **Training data:** DROID dataset (76,000+ trajectories, 50+ robot types)
- Prefix pass fills KV cache once, then N denoising steps per inference call

**[ INSERT: Architecture diagram image here ]**

**Script:**
> "We enabled both models. OpenVLA uses a 7B Llama backbone and outputs discrete action tokens — simpler to export but larger. π0.5 is 3B parameters with a flow-matching denoiser that produces continuous, high-precision trajectories. The denoiser runs N times per call, which creates a special challenge for optimization — more on that shortly."

---

## Slide 7 — Enabling on Arc Pro B70

**XPU Patch Summary — minimal changes to upstream openpi and OpenVLA repos:**

### π0.5 (openpi)
| File | Change |
|------|--------|
| `pyproject.toml` | torch 2.10.0+xpu; remove jax[cuda12] → jax 0.5.3; add pytorch-xpu index |
| `policy_config.py` | XPU auto-detection: xpu > cuda > cpu fallback |
| `examples/libero/requirements.txt` | Comment out CUDA torch (conflicts with XPU venv) |
| `third_party/libero/*` | Add `__init__.py`; fix `torch.load weights_only=False` (submodule) |

### OpenVLA
| File | Change |
|------|--------|
| `requirements.txt` | torch 2.10.0+xpu; add pytorch-xpu index |
| `vla/modeling_prismatic.py` | XPU device map; replace `.cuda()` calls with `.to(device)` |
| `inference.py` | Auto-detect XPU; remove CUDA-specific dtype assertions |

**Note on subprocess architecture:** Loading the 3B model onto XPU installs process-wide Level-Zero memory hooks that conflict with MuJoCo's osmesa CPU allocator → segfault. Fix: LIBERO simulator runs in a clean worker subprocess that never imports torch. Communication via pickle IPC over stdin/stdout.

**Script:**
> "The changes were surprisingly minimal. For π0.5, the four patches shown here took about a day. OpenVLA required a few more changes because it had more hardcoded CUDA assumptions. The trickiest issue was a segfault from Level-Zero memory hooks conflicting with MuJoCo — solved by isolating the simulator in a subprocess."

---

## Slide 8 — LIBERO Benchmark

**What is LIBERO?**
- A standardized benchmark for robot manipulation learning built on **MuJoCo** physics simulation
- Tests generalization across four dimensions: spatial reasoning, object recognition, goal understanding, and long-horizon planning

**Four task suites (10 tasks each, 5 trials per task):**
| Suite | Tests |
|-------|-------|
| LIBERO-Spatial | Same objects, different positions — tests spatial reasoning |
| LIBERO-Object | Same positions, different objects — tests object recognition |
| LIBERO-Goal | Same scene, different instructions — tests language understanding |
| LIBERO-10 | Long-horizon tasks requiring 3+ sub-goals in sequence |

**How it works:**
- Robot arm in a tabletop scene with camera observations
- Model receives: 3 camera frames (224×224) + natural language instruction
- Outputs: 7-DOF joint velocities for a 10-step action horizon
- Episode succeeds when task goal is achieved within 600 timesteps

**[ INSERT: LIBERO simulation screenshot / rollout image here ]**

**Script:**
> "LIBERO is the standard simulation benchmark for manipulation policies. It uses MuJoCo physics, which is why we ran into the subprocess issue. The four suites systematically test different kinds of generalization. We ran 5 trials per task across all 10 tasks per suite — 200 episodes total per configuration. This is what Physical Intelligence used to validate π0.5 originally."

---

## Slide 9 — Benchmark Results

**B70 vs RTX PRO 4000 — both running PyTorch, then B70 with OpenVINO:**

### DROID inference (2 cameras, 15-step horizon, 10 denoising steps)
| Configuration | Latency | Hz |
|---|---|---|
| B70 · PyTorch XPU | 118.7 ms | 8.4 Hz |
| RTX PRO 4000 · PyTorch | 83.6 ms | 12.0 Hz |
| **B70 · OpenVINO FP32** | **88.3 ms** | **11.3 Hz** |

### LIBERO (4 suites, 10 denoising steps)
| Suite | B70 · PyTorch | RTX PRO 4000 · PyTorch | B70 · OpenVINO FP32 | Success (OV) |
|---|---|---|---|---|
| LIBERO-Spatial | 129.9 ms | 114.8 ms | **112.6 ms** ✓ | 100% |
| LIBERO-Object | 130.4 ms | 115.1 ms | **110.5 ms** ✓ | 98% |
| LIBERO-Goal | 130.2 ms | 114.9 ms | **111.6 ms** ✓ | 96% |
| LIBERO-10 | 130.1 ms | 115.1 ms | **110.3 ms** ✓ | 90% |

**Key result:** B70 + OpenVINO FP32 beats RTX PRO 4000 PyTorch across all four LIBERO suites — at **$949 vs $1,699**.

**Script:**
> "Out of the box with PyTorch XPU, the B70 trails the RTX PRO 4000 by about 30 ms on DROID. But once we bring in OpenVINO, the B70 closes that gap and actually beats the NVIDIA card on LIBERO — at nearly half the price. To understand why, let's look at what the profiler told us."

---

## Slide 10 — PyTorch Profiling

**How we profiled:**
We ran `torch.profiler` on both the B70 and the NVIDIA machine, capturing full op-level traces:

```bash
# Capture trace on B70 (XPU)
python scripts/pi0.5_profile.py --task droid --tag baseline_xpu --device xpu

# Capture trace on NVIDIA (CUDA)
python scripts/pi0.5_profile.py --task droid --tag baseline_nvidia --device cuda
```

Each run produces a `pt.trace.json` file — a Chrome trace format file viewable in [Perfetto UI](https://ui.perfetto.dev).

**We fed both traces into Claude and asked it to explain the differences.**

**Key insight Claude surfaced:**

> **XPU:** 0 `cudaGraphLaunch` calls — each of 4,793 kernels enqueued individually via Level Zero (`urEnqueueKernelLaunch`), ~9–10 µs each → **~44 ms CPU overhead**
>
> **NVIDIA:** 12 `cudaGraphLaunch` calls per iteration — all 5,044 kernels submitted as pre-recorded replay graphs → **~7 ms CPU overhead**

The GPU compute times are nearly identical (79.4 ms B70 vs 76.3 ms NVIDIA). The **entire 30+ ms gap is software overhead**, not hardware.

**[ INSERT: Perfetto screenshot — XPU trace showing individual kernel launches ]**

**[ INSERT: Perfetto screenshot — NVIDIA trace showing cudaGraphLaunch events ]**

**Script:**
> "We ran torch.profiler on both machines and got Chrome trace JSON files. Loading them into Perfetto gives a full op-level and kernel-level timeline. We shared both with Claude and asked it to compare them. The insight it found was striking: the GPU work times are nearly equal — the B70 is only 3 ms behind on raw compute. The entire gap came from CPU dispatch overhead. NVIDIA's torch.compile captures kernels into 12 CUDA Graphs and replays them in 12 calls. XPU has no equivalent — each of 4,793 kernels gets enqueued one at a time."

---

## Slide 11 — OpenVINO Optimization

**The idea: compile the whole model into one graph, call it once.**

OpenVINO's approach bypasses the per-kernel dispatch problem entirely:

```
PyTorch model  →  ONNX IR  →  OV IR (.xml + .bin)  →  Compiled GPU graph
```

1. **Export to ONNX** — the entire π0.5 inference (prefix + all 10 denoising steps **unrolled**) becomes one static computation graph. KV cache stays internal to the graph.
2. **Convert to OV IR** — `ov.convert_model()` translates ONNX ops to OpenVINO's internal representation, optimized for Arc GPU via the oneDNN backend.
3. **Compile once** — `core.compile_model()` on first run (~30–60 s). The compiled graph is a pre-planned sequence of GPU work that the Arc GPU executes in one shot.
4. **Infer** — `infer_req.infer(inputs)` — one Python call, no kernel-by-kernel dispatch, near-zero CPU overhead.

**Result:**
- PyTorch XPU: 117 ms (79 ms GPU + 38 ms CPU dispatch)
- OpenVINO FP32: ~88 ms (single compiled graph call)

**Script:**
> "OpenVINO solves the dispatch problem by compiling the entire model — including the unrolled 10-step denoising loop — into a single graph representation ahead of time. The Arc GPU plugin then executes that pre-compiled plan with a single API call. Instead of 4,793 individual kernel enqueues, you get one. That's where the 30 ms of CPU overhead disappears."

---

## Slide 12 — Breaking 100 ms (5-Step Denoising)

**Flow matching is robust to fewer denoising steps.**

Unlike diffusion models, flow-matching trajectories are nearly straight lines in latent space. Physical Intelligence's own ablations — and the BLURR paper — confirm that **5 steps ≈ 10 steps** on success rate across all LIBERO suites.

| Configuration | Latency | Hz | vs baseline |
|---|---|---|---|
| B70 · PyTorch · 10 steps | ~130 ms | 7.7 | baseline |
| RTX PRO 4000 · PyTorch · 10 steps | ~115 ms | 8.7 | −12% |
| B70 · OpenVINO FP32 · 10 steps | 88–111 ms | 9–11 | −15 to −26% |
| **B70 · OpenVINO FP32 · 5 steps** | **~55–70 ms ✓** | **14–18** | **−46 to −57%** |

The 5-step OV model is exported separately:
```bash
python scripts/convert_libero_openvino.py \
  --export-onnx --onnx-to-ov --num-steps 5 \
  --ov-fp32-dir profiler_output/libero_fp32_steps5
```

**Script:**
> "The final optimization is reducing denoising steps from 10 to 5. Because flow matching follows nearly straight paths, halving the steps barely affects quality — all four LIBERO suites maintained their success rates within 2 percentage points. But it cuts inference time roughly in half, pushing DROID to ~55 ms and LIBERO to ~70 ms. We're well under the 100 ms target at 14–18 Hz."

---

## Slide 13 — Live Demo

**Real-time MuJoCo inference on Arc Pro B70**

- π0.5 LIBERO-Spatial: "pick up the red block and place it on the plate"
- Running: Arc Pro B70 · OpenVINO FP32 · 9 Hz
- Task success: 100% across all 5 trials

**[ INSERT: Video or screenshot of rollout ]**

Commands:
```bash
source .venv/bin/activate
export MUJOCO_GL=osmesa && export NUMBA_DISABLE_JIT=1

# OpenVINO 5-step — under 100 ms
python run_libero_xpu.py \
  --args.ov-model-path profiler_output/libero_fp32_steps5/model.xml \
  --args.num-steps 5 --args.task-suite-name all
```

Web viewer streams the simulation live at `http://localhost:9000`.

**Script:**
> "Here's the system running live. The robot is receiving camera frames and language instructions, running the full π0.5 inference on the B70 via OpenVINO, and sending actions back to MuJoCo at ~9 Hz for the 10-step model, or ~14 Hz for the 5-step model. 100% success on LIBERO-Spatial — it consistently picks up and places the objects correctly."

---

## Slide 14 — Q&A

**Summary numbers:**
- **−26%** DROID latency: PyTorch XPU → OpenVINO
- **11.3 Hz** DROID throughput on B70 · OV FP32
- **$949** Arc Pro B70 vs $1,699 RTX PRO 4000

---

## Notes for Presenter

### Image placeholders to fill in:
- **Slide 5:** Architecture diagram showing SigLIP + PaliGemma/Llama + action head for both models
- **Slide 7:** LIBERO MuJoCo simulation screenshot
- **Slide 9:** Two Perfetto screenshots — one showing dense individual kernel launches on XPU, one showing 12 `cudaGraphLaunch` blocks on NVIDIA
- **Slide 12:** Demo video or rollout screenshot

### How to open Perfetto traces:
1. Go to https://ui.perfetto.dev
2. Click "Open trace file"
3. Load `profiler_output/baseline_xpu/*.pt.trace.json` or `profiler_output/baseline_nvidia/*.pt.trace.json`
4. Search for `cudaGraphLaunch` (press `/`) to find the 12 NVIDIA graph launch events
5. Compare with XPU timeline showing dense individual `urEnqueueKernelLaunch` bars

### Key talking points for Q&A:
- "Why not use CUDA Graphs on XPU?" — PyTorch's XPU backend doesn't implement CUDA Graph capture; it's a roadmap item
- "Is OpenVINO numerically equivalent?" — Yes, MSE < 1e-4 vs PyTorch on validation inputs
- "Does 5-step work for all tasks?" — All four LIBERO suites within 2% success rate of 10-step; Physical Intelligence validated this
- "What about OpenVLA on B70?" — Enabled with 3 file patches; performance pending full benchmark run
