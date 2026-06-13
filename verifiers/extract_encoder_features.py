"""Post-hoc extraction of π₀ encoder features for each logged chunk (DAY 10, REMOTE).

The Day 6a/7 scripts saved (chunks, states_at_chunk, success) per episode.
They did NOT save images or encoder features. This script re-creates each
episode's env, replays to each chunk-issue point, captures the observation,
runs it through π₀'s encoder, and appends `obs_features` (N, ~520) to each npz.

Designed to run on the remote 4090 (needs pi0 loaded). Idempotent: skips files
that already have obs_features.

Usage (on remote):
    cd ~/flare/openpi
    PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 \\
      uv run python -u ~/flare/verifiers/extract_encoder_features.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path.home() / "flare"))

import numpy as np
import torch
from openpi_client import image_tools
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training import config as _c
from openpi.policies import policy_config
from patch_pi0_sde import apply_patch

LIBERO_DUMMY_ACTION = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
NUM_STEPS_WAIT = 10
MAX_STEPS_MAP = {"libero_10": 520}
REPLAN_STEPS = 5
CHECKPOINT_DIR = str(Path.home() / "flare/checkpoints/pi0_libero_pt")
TASK_SUITE = "libero_10"


def _quat2axisangle(q):
    q = np.array(q, dtype=np.float64)
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def format_obs(o, prompt, sz=224):
    img = np.ascontiguousarray(o["agentview_image"][::-1, ::-1])
    wimg = np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, sz, sz))
    wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, sz, sz))
    state = np.concatenate([
        o["robot0_eef_pos"].astype(np.float32),
        _quat2axisangle(o["robot0_eef_quat"]),
        o["robot0_gripper_qpos"].astype(np.float32),
    ])
    return {"observation/image": img, "observation/wrist_image": wimg,
            "observation/state": state, "prompt": prompt}


_ENCODER_PATH = ("paligemma_with_expert", "paligemma", "model", "vision_tower")


def _get_encoder_module(model):
    """Walk the nested path to π₀'s SigLIP vision tower."""
    obj = model
    for attr in _ENCODER_PATH:
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"Could not find encoder at {'.'.join(_ENCODER_PATH)}; "
                f"failed at .{attr} on {type(obj).__name__}"
            )
        obj = getattr(obj, attr)
    return obj


def get_encoder_features(policy, model, example):
    """Run π₀ encoder forward and return pooled features.

    Hooks SiglipVisionModel (model.paligemma_with_expert.paligemma.model.vision_tower).
    π₀ passes BOTH agentview and wrist images through the vision tower, so the
    hook fires twice — we concatenate the two pooled features.
    """
    cached = {"feats": []}

    def hook(module, inp, out):
        # Handle HuggingFace output formats: tuple, ModelOutput, or raw tensor
        if isinstance(out, tuple):
            x = out[0]
        elif hasattr(out, "last_hidden_state"):
            x = out.last_hidden_state
        else:
            x = out
        # x shape: (B, num_patches, hidden_dim). Mean-pool over patches.
        if x.ndim == 3:
            x = x.mean(dim=1)
        feat = x.detach().cpu().float().numpy().squeeze()
        cached["feats"].append(feat)

    target = _get_encoder_module(model)
    handle = target.register_forward_hook(hook)
    try:
        _ = policy.infer(example)
    finally:
        handle.remove()

    if not cached["feats"]:
        return np.zeros(1152, dtype=np.float32)  # fallback (SigLIP-So400m hidden dim)

    # Concatenate features from all hook fires (typically 2: agentview + wrist)
    feat = np.concatenate(cached["feats"], axis=-1).astype(np.float32)
    return feat


def replay_to_chunk(env, init_state, n_chunks_executed: int):
    """Step through env: dummy actions → n_chunks_executed chunks (using their
    actual saved actions), returning the obs at the next chunk-issue point.

    For accurate replay we'd need the actual executed action sequence. Since we
    have the chunks themselves saved, we can replay them:
        - reset env, set_init_state(init)
        - dummy NUM_STEPS_WAIT
        - for i in 0..n_chunks_executed-1: step through first REPLAN_STEPS actions of chunks[i]
        - return obs (this is the obs the (n_chunks_executed)-th chunk was issued at)
    """
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def process_one_npz(npz_path: Path, policy, model, task, init_state):
    """Re-create env, walk through chunks, extract encoder features, augment npz."""
    data = dict(np.load(npz_path))
    if "obs_features" in data:
        # Skip only if features are NON-zero (the previous broken extraction
        # wrote all-zero features; those need re-processing)
        if not np.allclose(data["obs_features"], 0):
            return False
        # else: drop the all-zero placeholder and re-extract
        del data["obs_features"]
    chunks = data["chunks"]
    N = chunks.shape[0]
    if N == 0:
        return False

    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256
    )
    env.seed(42)
    features = []

    try:
        obs = replay_to_chunk(env, init_state, 0)
        for t in range(N):
            example = format_obs(obs, task.language)
            feat = get_encoder_features(policy, model, example)
            features.append(feat)
            # Execute first REPLAN_STEPS actions of chunks[t]
            chunk = chunks[t]
            for a in chunk[:REPLAN_STEPS]:
                obs, _, done, _ = env.step(np.array(a, dtype=np.float32))
                if done:
                    break
            if done:
                # Pad remaining features with zeros
                while len(features) < N:
                    features.append(np.zeros_like(feat))
                break
    finally:
        env.close()

    obs_features = np.stack(features).astype(np.float32)
    data["obs_features"] = obs_features
    np.savez_compressed(npz_path, **data)
    # Sanity tag so main() can print stats on the first successful file
    process_one_npz._last_feat_stats = (
        obs_features.shape,
        float(obs_features.mean()),
        float(obs_features.std()),
        bool(np.allclose(obs_features, 0)),
    )
    return True


def main():
    print("Loading policy (post-patch)...")
    Pi0 = apply_patch()
    cfg = _c.get_config("pi0_libero")
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT_DIR)
    model = None
    for attr in ["_model", "model", "_policy", "policy"]:
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            print(f"  found model under policy.{attr}")
            break
    if model is None:
        raise RuntimeError("Could not find Pi0 instance")

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

    # Process Day 6a chunks
    day6a_chunks = Path.home() / "flare/results/day6a_sde_sweep/chunks"
    day7_chunks = Path.home() / "flare/results/day7_outcome_scan/chunks"

    paths = []
    if day6a_chunks.exists():
        paths.extend(sorted(day6a_chunks.glob("eta_*.npz")))
    if day7_chunks.exists():
        paths.extend(sorted(day7_chunks.glob("eta_*.npz")))

    print(f"Processing {len(paths)} npz files...")
    t0 = time.time()
    done_count = 0
    skipped = 0
    for i, p in enumerate(paths):
        # Parse filename to get task + init
        stem = p.stem
        parts = stem.split("_")
        try:
            task_id = int(parts[2][1:])
            if parts[3].startswith("ep"):
                init_idx = int(parts[3][2:])
            else:
                init_idx = int(parts[3][4:])
        except (IndexError, ValueError):
            print(f"  skipping unparseable: {p.name}")
            continue
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        if init_idx >= len(init_states):
            print(f"  skipping (init_idx OOB): {p.name}")
            continue
        try:
            ran = process_one_npz(p, policy, model, task, init_states[init_idx])
            if ran:
                done_count += 1
                # Print sanity stats on the first successful extraction
                if done_count == 1 and hasattr(process_one_npz, "_last_feat_stats"):
                    shape, mean, std, all_zero = process_one_npz._last_feat_stats
                    print(f"  >>> FIRST FEATURES: shape={shape} mean={mean:.4f} "
                          f"std={std:.4f} all_zero={all_zero}", flush=True)
                    if all_zero:
                        raise RuntimeError(
                            "First file extracted ALL-ZERO features — hook still wrong. "
                            "Aborting before more files get poisoned."
                        )
            else:
                skipped += 1
        except Exception as e:
            print(f"  error on {p.name}: {e}")
            continue
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(paths) - i - 1) / max(rate, 0.001)
            print(
                f"  [{i+1:4d}/{len(paths)}] done={done_count} skipped={skipped} "
                f"rate={rate:.1f}/s eta={eta:.0f}s"
            )

    print(f"\nDone: processed {done_count}, skipped {skipped} (already done), "
          f"total {len(paths)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
