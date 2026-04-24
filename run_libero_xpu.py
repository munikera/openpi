"""
OpenPI LIBERO Evaluation on Intel XPU — Single Command
=========================        # Action un-normalization: model outputs in normalized space → robot space.
        # pi05_libero uses USE_QUANTILE_NORM=True, so the formula is QUANTILE un-normalization:
        #   x = (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        # This is implemented by transforms.Unnormalize(use_quantiles=True).
        if "actions" in norm_stats:
            ns = norm_stats["actions"]
            if ns.q01 is not None and ns.q99 is not None:
                self._action_q01 = np.array(ns.q01, dtype=np.float32)[:7]
                self._action_q99 = np.array(ns.q99, dtype=np.float32)[:7]
                print(f"[OVLiberoPolicy] Using QUANTILE unnorm for actions")
                print(f"[OVLiberoPolicy] action q01[:3]: {self._action_q01[:3]}")
                print(f"[OVLiberoPolicy] action q99[:3]: {self._action_q99[:3]}")
            else:
                # Fallback to z-score (not expected for pi05)
                self._action_q01 = None
                self._action_q99 = None
                self._action_mean = np.array(ns.mean, dtype=np.float32)[:7]
                self._action_std  = np.array(ns.std,  dtype=np.float32)[:7]
                print(f"[OVLiberoPolicy] WARNING: using z-score unnorm (no q01/q99 found)")
        else:
            self._action_q01 = None
            self._action_q99 = None
            print("[OVLiberoPolicy] WARNING: no 'actions' norm_stats — outputs NOT unnormalized")==================

Based on examples/libero/main.py but loads the Pi0.5 model DIRECTLY on XPU
in the main process and runs LIBERO simulation in a subprocess worker
(libero_env_worker.py — no torch imported). No separate server needed.

Architecture (same pattern as OpenVLA's libero_env_worker.py):
  Main process:  Pi0.5 model on XPU → inference
  Subprocess:    libero_env_worker.py → LIBERO simulation (no torch/osmesa conflict)

Usage:
  export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"
  export MUJOCO_GL=osmesa
  export NUMBA_DISABLE_JIT=1
  uv run python run_libero_xpu.py
  uv run python run_libero_xpu.py --args.task-suite-name all
  uv run python run_libero_xpu.py --args.task-suite-name libero_spatial --args.num-trials-per-task 50
"""

import base64
import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import pickle
import subprocess
import sys
import time
from datetime import datetime

import imageio
import numpy as np
import tqdm
import tyro

from web_viewer import WebViewer
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
WORKER_SCRIPT = pathlib.Path(__file__).parent / "libero_env_worker.py"


# ── OpenVINO policy wrapper ───────────────────────────────────────────────

class OVLiberoPolicy:
    """Drop-in replacement for a PyTorch policy that uses a compiled OV model.

    Replicates the preprocessing from LiberoInputs + the model's
    _preprocess_observation, then calls OV infer instead of torch forward.

    The OV model was exported from LiberoInferenceWrapper, which:
      - Takes (images[1,3,3,224,224], img_masks[1,3], lang_tokens[1,48],
               lang_masks[1,48], state[1,8], noise[1,10,32])
      - Returns actions[1,10,7]  (in NORMALIZED model space)

    Transforms applied to match the PyTorch policy pipeline:
      Input:  images uint8 [H,W,C] → float32 CHW [-1, 1]  (Observation.from_dict)
      Output: actions (model space) → robot space via Unnormalize(norm_stats)
      Noise:  N(0,1) random, same as sample_noise()
    """

    def __init__(self, xml_path: str, tokenizer, norm_stats: dict, ov_device: str = "GPU",
                 num_steps: int = 10, action_horizon: int = 10,
                 action_dim_full: int = 32, tokenizer_len: int = 48,
                 state_dim: int = 8):
        import openvino as ov
        from openpi_client import image_tools as _it

        self._it = _it
        self.num_steps = num_steps
        self.action_horizon = action_horizon
        self.action_dim_full = action_dim_full
        self.tokenizer_len = tokenizer_len
        self.state_dim = state_dim
        self._tokenizer = tokenizer

        # Action un-normalization: model outputs in normalized space → robot space
        # pi05_libero uses USE_QUANTILE_NORM=True, so the formula is QUANTILE un-normalization:
        #   x = (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        # This is implemented by transforms.Unnormalize(use_quantiles=True).
        if "actions" in norm_stats:
            ns = norm_stats["actions"]
            if ns.q01 is not None and ns.q99 is not None:
                self._action_q01 = np.array(ns.q01, dtype=np.float32)[:7]
                self._action_q99 = np.array(ns.q99, dtype=np.float32)[:7]
                print(f"[OVLiberoPolicy] Using QUANTILE unnorm for actions")
                print(f"[OVLiberoPolicy] action q01[:3]: {self._action_q01[:3]}")
                print(f"[OVLiberoPolicy] action q99[:3]: {self._action_q99[:3]}")
            else:
                # Fallback to z-score (not expected for pi05)
                self._action_q01 = None
                self._action_q99 = None
                self._action_mean = np.array(ns.mean, dtype=np.float32)[:7]
                self._action_std  = np.array(ns.std,  dtype=np.float32)[:7]
                print(f"[OVLiberoPolicy] WARNING: using z-score unnorm (no q01/q99 found)")
        else:
            self._action_q01 = None
            self._action_q99 = None
            print("[OVLiberoPolicy] WARNING: no 'actions' norm_stats — outputs NOT unnormalized")

        core = ov.Core()
        config = {}
        if ov_device.startswith("GPU"):
            config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"
        compiled = core.compile_model(xml_path, device_name=ov_device, config=config)
        self._infer_req = compiled.create_infer_request()
        print(f"[OVLiberoPolicy] Compiled on {ov_device}: {xml_path}")

    def _tokenize(self, prompt: str):
        # PaligemmaTokenizer.tokenize() returns (tokens_np[max_len], mask_np[max_len])
        tokens, masks = self._tokenizer.tokenize(prompt)
        return tokens.astype(np.int64), masks.astype(np.float32)

    def infer(self, obs: dict) -> dict:
        """obs keys: observation/image, observation/wrist_image, observation/state, prompt"""
        img = obs["observation/image"]       # [224,224,3] uint8
        wrist = obs["observation/wrist_image"]  # [224,224,3] uint8
        state = np.asarray(obs["observation/state"], dtype=np.float32)

        # Images must be float32 in [-1, 1] CHW — matching Observation.from_dict which does:
        #   uint8 [B,H,W,C]  →  float32 [B,C,H,W]  /255 * 2 - 1
        # LiberoInferenceWrapper calls embed_prefix directly (bypassing _preprocess_observation),
        # so the OV model was exported expecting [-1, 1] float32 CHW inputs.
        img_f   = img.astype(np.float32) / 255.0 * 2.0 - 1.0    # [224,224,3]  values [-1, 1]
        wrist_f = wrist.astype(np.float32) / 255.0 * 2.0 - 1.0  # [224,224,3]  values [-1, 1]
        zeros_f = np.zeros_like(img_f)                            # right_wrist = zeros (masked out)

        # Stack cameras: [1, 3, 3, 224, 224]  (CHW order for the model)
        def hwc_to_chw(x):
            return np.transpose(x, (2, 0, 1))

        images = np.stack([
            hwc_to_chw(img_f),
            hwc_to_chw(wrist_f),
            hwc_to_chw(zeros_f),
        ], axis=0)[None]  # [1,3,3,224,224]

        img_masks = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # right_wrist masked

        lang_tokens, lang_masks = self._tokenize(str(obs.get("prompt", "")))
        lang_tokens = lang_tokens[None]  # [1, 48]
        lang_masks  = lang_masks[None]   # [1, 48]

        state_in = state[None]  # [1, 8]

        # Sample noise from N(0,1) — same as model.sample_noise() in the PyTorch path.
        # Using zeros would always start denoising from the same point → wrong trajectories.
        noise = np.random.randn(1, self.action_horizon, self.action_dim_full).astype(np.float32)

        inputs = [images, img_masks, lang_tokens, lang_masks, state_in, noise]
        self._infer_req.infer(inputs)
        actions = self._infer_req.get_output_tensor(0).data.copy()  # [1, 10, 7]
        actions = actions[0]  # [10, 7]

        # Un-normalize actions from model space → robot space.
        # pi05_libero uses quantile normalization (use_quantile_norm=True):
        #   model output x_norm ∈ [-1, 1]  →  robot space: (x_norm + 1) / 2 * (q99 - q01) + q01
        if self._action_q01 is not None:
            actions = (actions + 1.0) / 2.0 * (self._action_q99 - self._action_q01 + 1e-6) + self._action_q01
        elif hasattr(self, "_action_mean") and self._action_mean is not None:
            # fallback z-score (not used for pi05)
            actions = actions * (self._action_std + 1e-6) + self._action_mean

        return {"actions": actions}  # [10, 7]


# ── IPC helpers ──────────────────────────────────────────────────────────

def _encode(obj):
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

def _decode(line):
    return pickle.loads(base64.b64decode(line.strip()))


class EnvWorker:
    """Manages a LIBERO env running in a subprocess (no torch)."""

    def __init__(self):
        env = os.environ.copy()
        env["MUJOCO_GL"] = "osmesa"
        env["NUMBA_DISABLE_JIT"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
            bufsize=0,
        )

    def _send(self, msg):
        self._proc.stdin.write((_encode(msg) + "\n").encode())
        self._proc.stdin.flush()

    def _recv(self):
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker process died unexpectedly")
        resp = _decode(line)
        if resp.get("status") == "error":
            raise RuntimeError(f"Worker error: {resp['message']}")
        return resp

    def init_task(self, task_suite_name, task_id, resolution, seed):
        self._send({
            "cmd": "init",
            "task_suite_name": task_suite_name,
            "task_id": task_id,
            "resolution": resolution,
            "seed": seed,
        })
        resp = self._recv()
        self._current_task_id = task_id
        return resp["task_description"], resp["n_init_states"]

    def reset(self, init_state_idx):
        self._send({
            "cmd": "reset",
            "task_id": self._current_task_id,
            "init_state_idx": init_state_idx,
        })
        return self._recv()["obs"]

    def step(self, action):
        self._send({"cmd": "step", "action": action})
        resp = self._recv()
        return resp["obs"], resp["reward"], resp["done"]

    def close_env(self):
        self._send({"cmd": "close"})
        self._recv()

    def shutdown(self):
        try:
            self._send({"cmd": "shutdown"})
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


# ── Tee stream for log capture ───────────────────────────────────────────

class TeeStream:
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


# ── Args (mirrors examples/libero/main.py) ──────────────────────────────

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # OpenVINO: set to path of exported model.xml to use OV inference instead of PyTorch.
    # Export first with:  python scripts/convert_libero_openvino.py --export-onnx --onnx-to-ov
    # Example: ov_model_path = "profiler_output/libero_fp32/model.xml"
    #################################################################################################################
    ov_model_path: str | None = None  # None = use PyTorch; path = use OV GPU inference
    ov_device: str = "GPU"            # OV device: "GPU"=Arc, "CPU"=fallback, "AUTO"=best

    #################################################################################################################
    # Model parameters (new — not in main.py)
    #################################################################################################################
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    device: str = "xpu"  # "xpu", "cpu"

    #################################################################################################################
    # Model server parameters (kept from main.py for compatibility)
    #################################################################################################################
    resize_size: int = 224
    replan_steps: int = 5

    # Denoising steps for the flow-matching sampler.
    # Default (None) uses the model config value (10 for π0.5).
    # BLURR paper shows 4 steps ≈ same success rate at ~57% lower latency.
    num_steps: int | None = None

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all
    )
    num_steps_wait: int = 2  # Number of steps to wait for objects to stabilize in sim (must match main.py=10)
    num_trials_per_task: int = 5  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)

    # Web viewer: stream live simulation frames to http://localhost:<web_viewer_port>
    # Default is 0 (disabled). Pass --args.web-viewer-port 9000 to enable.
    web_viewer_port: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ── Main ─────────────────────────────────────────────────────────────────

def eval_libero(args: Args) -> None:
    ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    if args.task_suite_name == "all":
        suites = ALL_SUITES
    else:
        suites = [args.task_suite_name]

    # Load model ONCE on XPU
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    print(f"Loading model: config={args.config_name}, device={args.device}")
    print(f"Checkpoint: {checkpoint_dir}")
    config = _config.get_config(args.config_name)
    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None
    if sample_kwargs:
        print(f"Denoising steps: {args.num_steps}  (default is 10)")
    policy = _policy_config.create_trained_policy(
        config, checkpoint_dir, pytorch_device=args.device,
        sample_kwargs=sample_kwargs,
    )
    print("Model loaded.")

    # Load norm_stats (needed for OV action un-normalization)
    from openpi.training import checkpoints as _checkpoints
    from openpi.shared import download as _download
    _ckpt_dir = _download.maybe_download(checkpoint_dir)
    _data_config = config.data.create(config.assets_dirs, config.model)
    _norm_stats = _checkpoints.load_norm_stats(
        os.path.join(_ckpt_dir, "assets"), _data_config.asset_id
    )

    # Optionally replace with OpenVINO compiled model
    ov_policy: OVLiberoPolicy | None = None
    if args.ov_model_path is not None:
        from openpi.models.tokenizer import PaligemmaTokenizer as _PaligemmaTokenizer
        _tokenizer = _PaligemmaTokenizer(max_len=48)  # matches tokenizer_len in OVLiberoPolicy
        _num_steps = args.num_steps if args.num_steps is not None else 10
        ov_policy = OVLiberoPolicy(
            xml_path=args.ov_model_path,
            tokenizer=_tokenizer,
            norm_stats=_norm_stats,
            ov_device=args.ov_device,
            num_steps=_num_steps,
        )
        print(f"[OV] OpenVINO policy active on {args.ov_device}")

    # Start web viewer
    viewer: WebViewer | None = None
    if args.web_viewer_port > 0:
        viewer = WebViewer(port=args.web_viewer_port)
        viewer.start()

    # Warmup
    print("Warming up (3 iterations)...")
    from openpi.policies import libero_policy
    for _ in range(3):
        dummy = libero_policy.make_libero_example()
        dummy["prompt"] = "warmup"
        if ov_policy is not None:
            ov_policy.infer(dummy)
        else:
            policy.infer(dummy)
    print("Warmup done.\n")

    suite_summaries: list[tuple[str, float]] = []

    for suite_name in suites:
        suite_args = dataclasses.replace(args, task_suite_name=suite_name)
        sr = _eval_single_suite(suite_args, policy, viewer=viewer, ov_policy=ov_policy)
        suite_summaries.append((suite_name, sr))

    if viewer:
        viewer.stop()

    if len(suite_summaries) > 1:
        avg_sr = sum(sr for _, sr in suite_summaries) / len(suite_summaries)
        print("\n" + "=" * 90)
        print("COMBINED RESULTS - ALL SUITES")
        print(f"  Checkpoint : {args.checkpoint_dir}")
        print(f"  Device     : {args.device}")
        print("=" * 90)
        print(f"  {'Suite':<30s} {'Success Rate':>12s}")
        print("-" * 90)
        for name, sr in suite_summaries:
            print(f"  {name:<30s} {sr:11.1f}%")
        print("-" * 90)
        print(f"  {'AVERAGE':<30s} {avg_sr:11.1f}%")
        print("=" * 90)

        results_dir = pathlib.Path(args.video_out_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        combined = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint_dir,
            "device": args.device,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "suites": {name: sr for name, sr in suite_summaries},
            "average_success_rate": round(avg_sr, 2),
        }
        ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        combined_path = results_dir / f"combined_results_{ts}.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        print(f"\nCombined results saved to {combined_path}")


def _eval_single_suite(args: Args, policy, viewer: "WebViewer | None" = None,
                       ov_policy: "OVLiberoPolicy | None" = None) -> float:
    """Evaluate a single task suite. Returns success rate percentage (0-100)."""
    # Set random seed
    np.random.seed(args.seed)

    # Set up log file to capture all terminal output
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    log_ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_path = pathlib.Path(args.video_out_path) / f"run_{args.task_suite_name}_{log_ts}.log"
    log_fh = open(log_path, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_stdout, log_fh)
    sys.stderr = TeeStream(old_stderr, log_fh)
    logging.getLogger().addHandler(logging.StreamHandler(log_fh))
    print(f"Logging to {log_path}")

    # Get number of tasks
    from libero.libero import benchmark as _benchmark
    benchmark_dict = _benchmark.get_benchmark_dict()
    task_suite_meta = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite_meta.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_step_times = []  # raw t_i for every inference call across all episodes
    bench_start_time = time.time()
    task_results_json = []  # for JSON output

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Spawn a fresh worker for each task (clean memory, no torch)
        worker = EnvWorker()
        task_description, n_init_states = worker.init_task(
            args.task_suite_name, task_id, LIBERO_ENV_RESOLUTION, args.seed
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_step_times = []  # raw t_i for all inference calls in this task
        task_episode_avg_ms = []  # per-episode avg inference ms
        episode_records = []

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment via worker
            obs = worker.reset(episode_idx)
            action_plan = collections.deque()

            # Setup
            t = 0
            replay_images = []
            episode_inf_time = 0.0
            episode_inf_calls = 0
            episode_steps = 0
            ep_step_times = []  # raw t_i for this episode
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < args.num_steps_wait:
                        obs, reward, done = worker.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image (rotate 180° to match training)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Stream to web viewer
                    if viewer is not None:
                        viewer.push_frame(img)
                        viewer.push_wrist(wrist_img)
                        viewer.update_stats(
                            task=str(task_description),
                            episode=task_episodes + 1,
                            successes=task_successes,
                            total=task_episodes + 1,
                        )

                    if not action_plan:
                        # Compute new action chunk — direct model inference (no server)
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        infer_start = time.time()
                        if ov_policy is not None:
                            result = ov_policy.infer(element)
                        else:
                            result = policy.infer(element)
                        action_chunk = result["actions"]
                        infer_end = time.time()

                        inf_dt = infer_end - infer_start
                        episode_inf_calls += 1
                        episode_inf_time += inf_dt
                        ep_step_times.append(inf_dt)
                        all_step_times.append(inf_dt)

                        # Update web viewer inference FPS (based on true model latency)
                        if viewer is not None and inf_dt > 0:
                            viewer._update_fps(inf_dt)

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    episode_steps += 1

                    # Execute action in environment (via worker)
                    obs, reward, done = worker.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Per-episode average: ep_avg = sum(inf times) / num inference calls
            ep_avg_ms = (episode_inf_time / episode_inf_calls * 1000) if episode_inf_calls > 0 else 0.0
            task_step_times.extend(ep_step_times)
            task_episode_avg_ms.append(ep_avg_ms)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            if replay_images:
                imageio.mimwrite(
                    str(video_path),
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            episode_records.append({
                "episode": episode_idx,
                "success": bool(done),
                "steps": episode_steps,
                "inf_calls": episode_inf_calls,
                "ep_avg_inf_ms": round(ep_avg_ms, 1),
                "ep_inf_time_s": round(episode_inf_time, 2),
                "video": str(video_path) if replay_images else "",
            })

            # Log current results
            logging.info(f"Success: {done} | steps: {episode_steps} | inf_calls: {episode_inf_calls} | avg_inf: {ep_avg_ms:.1f} ms | total_inf: {episode_inf_time:.1f}s")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log task results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # Per task avg: mean of per-episode averages
        task_avg_ms = float(np.mean(task_episode_avg_ms)) if task_episode_avg_ms else 0.0

        task_results_json.append({
            "description": task_description,
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": round(float(task_successes) / float(task_episodes) * 100, 2) if task_episodes > 0 else 0,
            "total_inf_calls": len(task_step_times),
            "task_avg_inf_ms": round(task_avg_ms, 2),
            "total_inf_time_s": round(sum(task_step_times), 2),
            "episode_results": episode_records,
        })

        worker.shutdown()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # ── Benchmark Summary ────────────────────────────────────────────────
    # Inference time hierarchy:
    #   Per step:          t_i = time() after - before (raw latency of one model call)
    #   Per episode avg:   ep_avg = mean(ep_step_times)   → avg step latency for this episode
    #   Per episode total: ep_inf_s = sum(ep_step_times)   → true total time for this episode
    #   Per task avg:      mean(per-episode averages)      → avg step latency over all episodes
    #   Per task total:    sum(all step times in task)      → accurate total
    bench_elapsed = time.time() - bench_start_time
    success_rate = float(total_successes) / float(total_episodes) * 100.0

    # Global avg: mean of all per-episode averages
    all_ep_avgs = [ep["ep_avg_inf_ms"] for t in task_results_json for ep in t["episode_results"]]
    global_avg_ms = float(np.mean(all_ep_avgs)) if all_ep_avgs else 0.0
    global_hz = 1000.0 / global_avg_ms if global_avg_ms > 0 else 0.0
    total_inf_time = sum(t for t in all_step_times)

    if all_step_times:
        std_infer_ms = np.std(all_step_times) * 1000.0
        p50_infer_ms = np.percentile(all_step_times, 50) * 1000.0
        p90_infer_ms = np.percentile(all_step_times, 90) * 1000.0
        p99_infer_ms = np.percentile(all_step_times, 99) * 1000.0
    else:
        std_infer_ms = p50_infer_ms = p90_infer_ms = p99_infer_ms = 0.0

    suite_names = {
        "libero_spatial": "LIBERO-Spatial",
        "libero_object": "LIBERO-Object",
        "libero_goal": "LIBERO-Goal",
        "libero_10": "LIBERO-Long (libero_10)",
        "libero_90": "LIBERO-90",
    }
    suite_label = suite_names.get(args.task_suite_name, args.task_suite_name)

    print("\n" + "=" * 90)
    print("LIBERO BENCHMARK RESULTS")
    print("=" * 90)
    print(f"  Suite              : {suite_label}")
    print(f"  Device             : {args.device}")
    print(f"  Checkpoint         : {args.checkpoint_dir}")
    print(f"  Denoise steps      : {args.num_steps if args.num_steps is not None else 10} (default=10)")
    print(f"  Trials/Task        : {args.num_trials_per_task}")
    print(f"  Total Episodes     : {total_episodes}")
    print(f"  Success Rate       : {success_rate:.1f}%")
    print(f"  Total Wall Time    : {bench_elapsed:.0f}s ({bench_elapsed/60:.1f} min)")
    print(f"  Total Inf Time     : {total_inf_time:.0f}s ({total_inf_time/60:.1f} min)")
    print(f"  Inference Calls    : {len(all_step_times)}")
    print(f"  Avg Inference      : {global_avg_ms:.1f} ms/call  (mean of per-episode averages)")
    print(f"  Inference P50      : {p50_infer_ms:.1f} ms")
    print(f"  Inference P90      : {p90_infer_ms:.1f} ms")
    print(f"  Inference P99      : {p99_infer_ms:.1f} ms")
    print(f"  Inference Rate     : {global_hz:.2f} Hz")
    print("=" * 90)

    # Per-task table
    print(f"\n{'Task':<85s} {'SR':>6s}  {'Eps':>4s}  {'InfCalls':>8s}  {'TaskAvg':>9s}  {'InfTime':>8s}")
    print("-" * 120)
    for tr in task_results_json:
        sr = tr["success_rate"]
        print(
            f"{tr['description']:<85s} {sr:5.1f}%  {tr['episodes']:4d}  "
            f"{tr['total_inf_calls']:8d}  {tr['task_avg_inf_ms']:7.1f}ms  {tr['total_inf_time_s']:7.1f}s"
        )
    print("-" * 120)
    print(
        f"{'TOTAL':<85s} {success_rate:5.1f}%  {total_episodes:4d}  "
        f"{len(all_step_times):8d}  {global_avg_ms:7.1f}ms  {total_inf_time:7.1f}s"
    )
    print("=" * 120)

    # Machine-readable one-liner
    print(
        f"\nBENCHMARK_CSV: {suite_label},{success_rate:.1f}%,"
        f"{global_avg_ms:.1f}ms,{global_hz:.2f}Hz,"
        f"{bench_elapsed:.0f}s,{args.num_trials_per_task}"
    )

    # Save JSON results
    results_path = pathlib.Path(args.video_out_path) / f"results_{args.task_suite_name}_{log_ts}.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "task_suite": args.task_suite_name,
        "checkpoint": args.checkpoint_dir,
        "device": args.device,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": round(success_rate, 2),
        "total_wall_time_s": round(bench_elapsed, 1),
        "global_avg_inference_ms": round(float(global_avg_ms), 2),
        "global_inference_hz": round(float(global_hz), 2),
        "total_inf_calls": len(all_step_times),
        "total_inference_time_s": round(total_inf_time, 1),
        "p50_ms": round(float(p50_infer_ms), 2),
        "p90_ms": round(float(p90_infer_ms), 2),
        "p99_ms": round(float(p99_infer_ms), 2),
        "tasks": task_results_json,
    }
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_path}")

    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_fh.close()
    print(f"Full terminal log saved to {log_path}")

    return success_rate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
