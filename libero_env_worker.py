"""
LIBERO environment worker — runs in a subprocess with NO torch imported.

Communicates with the parent process via stdin/stdout using pickle over base64.
This avoids the segfault caused by torch/Level-Zero memory hooks conflicting
with osmesa's CPU allocator in OffScreenRenderEnv.

Protocol (line-based, base64-encoded pickle):
  Parent → Worker:
    {"cmd": "init", "task_suite_name": str, "task_id": int, "resolution": int, "seed": int}
    {"cmd": "reset", "init_state_idx": int}
    {"cmd": "step", "action": list[float]}
    {"cmd": "get_obs"}
    {"cmd": "close"}

  Worker → Parent:
    {"status": "ok", ...}  or  {"status": "error", "message": str}
"""

import base64
import os
import pickle
import sys
import pathlib
import math
import warnings

# ── CRITICAL: Capture the real stdout fd for IPC BEFORE anything can print ──
# gym, robosuite, and libero all print warnings to stdout at import time,
# which corrupts our base64 IPC protocol. We duplicate the real stdout fd,
# then redirect Python's sys.stdout to stderr so all warnings go there.
_IPC_FD = os.dup(sys.stdout.fileno())  # duplicate the real stdout fd
_IPC_WRITE = os.fdopen(_IPC_FD, "w")   # wrap in a Python file object
sys.stdout = sys.stderr                 # all future prints go to stderr

# Suppress the gym deprecation warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
os.environ["GYM_NO_DEPRECATION_WARNING"] = "1"

import numpy as np


def _encode(obj):
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _decode(line):
    return pickle.loads(base64.b64decode(line.strip()))


def _send(obj):
    _IPC_WRITE.write(_encode(obj) + "\n")
    _IPC_WRITE.flush()


def _recv():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return _decode(line)


def _quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    env = None
    task_suite = None
    resolution = 256

    while True:
        msg = _recv()
        cmd = msg["cmd"]

        try:
            if cmd == "init":
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict[msg["task_suite_name"]]()
                resolution = msg.get("resolution", 256)
                seed = msg.get("seed", 7)
                task_id = msg["task_id"]

                task = task_suite.get_task(task_id)
                task_description = task.language
                init_states = task_suite.get_task_init_states(task_id)

                bddl_file = (
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_file,
                    camera_heights=resolution,
                    camera_widths=resolution,
                )
                env.seed(seed)

                _send({
                    "status": "ok",
                    "task_description": task_description,
                    "n_init_states": len(init_states),
                })

            elif cmd == "reset":
                obs = env.reset()
                init_states = task_suite.get_task_init_states(msg.get("task_id", 0))
                obs = env.set_init_state(init_states[msg["init_state_idx"]])
                _send({
                    "status": "ok",
                    "obs": _extract_obs(obs),
                })

            elif cmd == "step":
                obs, reward, done, info = env.step(msg["action"])
                _send({
                    "status": "ok",
                    "obs": _extract_obs(obs),
                    "reward": float(reward),
                    "done": bool(done),
                })

            elif cmd == "get_obs":
                # Return current observation (already stored from last step/reset)
                pass

            elif cmd == "close":
                if env is not None:
                    env.close()
                    env = None
                _send({"status": "ok"})

            elif cmd == "shutdown":
                if env is not None:
                    env.close()
                _send({"status": "ok"})
                sys.exit(0)

            else:
                _send({"status": "error", "message": f"Unknown command: {cmd}"})

        except Exception as e:
            _send({"status": "error", "message": str(e)})


def _extract_obs(obs):
    """Extract the relevant observation fields as numpy arrays."""
    return {
        "agentview_image": np.ascontiguousarray(obs["agentview_image"]),
        "robot0_eye_in_hand_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
        "robot0_eef_pos": np.array(obs["robot0_eef_pos"], dtype=np.float64),
        "robot0_eef_quat": np.array(obs["robot0_eef_quat"], dtype=np.float64),
        "robot0_gripper_qpos": np.array(obs["robot0_gripper_qpos"], dtype=np.float64),
    }


if __name__ == "__main__":
    main()
