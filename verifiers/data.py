"""Chunk dataset loading and featurization.

Each training example:
    (obs_features, chunk_features, y_v3mc, y_wmt, task_id, episode_id, chunk_idx, n_chunks, eta, success)

Features:
  obs_features:
    mode="state"        → 8-d eef_pos(3) + axis_angle_quat(3) + gripper_qpos(2)
    mode="pi0_encoder"  → ~520-d pi0 encoder pool (requires post-hoc extraction)
  chunk_features:
    mode="pooled"       → 14-d: mean+std over 50 steps for each of 7 action dims
    mode="flattened"    → 350-d: full 50×7 sequence unrolled

Labels:
  y_v3mc: discounted MC label, γ^(T−t) · R_terminal  where R_terminal ∈ {0,1}
  y_wmt:  4-d (terminal_eef_pos[3], terminal_gripper_open_avg[1])
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ChunkExample:
    obs_features: np.ndarray
    chunk_features: np.ndarray
    y_v3mc: float
    y_wmt: np.ndarray  # (4,)
    task_id: int
    episode_id: str
    chunk_idx: int
    n_chunks: int
    eta: float
    success: bool


# -----------------------------------------------------------------------------
# Filename + JSON parsing
# -----------------------------------------------------------------------------
def parse_npz_filename(path: Path) -> dict:
    """Parse:
       Day 6a:  eta_<eta>_t<task>_ep<ep>.npz
       Day 7:   eta_<eta>_t<task>_init<init>_k<k>.npz
    """
    parts = path.stem.split("_")
    info: dict = {"path": path}
    if parts[0] != "eta":
        raise ValueError(f"Unexpected filename: {path.name}")
    info["eta"] = float(parts[1])
    info["task_id"] = int(parts[2][1:])
    if len(parts) == 4 and parts[3].startswith("ep"):
        info["episode_idx"] = int(parts[3][2:])
        info["init_idx"] = None
        info["k"] = None
        info["episode_id"] = (
            f"d6a_eta{info['eta']}_t{info['task_id']}_ep{info['episode_idx']}"
        )
    elif len(parts) == 5 and parts[3].startswith("init") and parts[4].startswith("k"):
        info["init_idx"] = int(parts[3][4:])
        info["k"] = int(parts[4][1:])
        info["episode_idx"] = None
        info["episode_id"] = (
            f"d7_eta{info['eta']}_t{info['task_id']}_"
            f"init{info['init_idx']}_k{info['k']}"
        )
    else:
        raise ValueError(f"Unrecognized filename format: {path.name}")
    return info


def _json_entry_to_episode_id(entry: dict, eta: float) -> str:
    if "episode_idx" in entry:
        return f"d6a_eta{eta}_t{entry['task_id']}_ep{entry['episode_idx']}"
    return (
        f"d7_eta{eta}_t{entry['task_id']}_"
        f"init{entry['init_idx']}_k{entry['k']}"
    )


def load_summary_jsons(day6a_dir: Path | str, day7_dir: Path | str) -> dict:
    """Index summary JSON entries by episode_id → terminal_state dict + success."""
    day6a = Path(day6a_dir)
    day7 = Path(day7_dir)
    index = {}
    for d in [day6a, day7]:
        if not d.exists():
            continue
        for json_path in sorted(d.glob("eta_*.json")):
            try:
                eta = float(json_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            try:
                data = json.load(open(json_path))
            except json.JSONDecodeError:
                print(f"  warn: could not parse {json_path}, skipping")
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                ep_id = _json_entry_to_episode_id(entry, eta)
                index[ep_id] = {
                    "terminal_state": entry.get("terminal_state"),
                    "success": entry.get("success", False),
                    "n_chunks": entry.get("n_chunks"),
                    "task_id": entry["task_id"],
                }
    return index


# -----------------------------------------------------------------------------
# Featurization
# -----------------------------------------------------------------------------
def featurize_chunk(
    chunk: np.ndarray, mode: Literal["pooled", "flattened"]
) -> np.ndarray:
    """chunk: (50, 7) → (14,) pooled or (350,) flattened."""
    if mode == "pooled":
        return np.concatenate([chunk.mean(axis=0), chunk.std(axis=0)]).astype(np.float32)
    elif mode == "flattened":
        return chunk.flatten().astype(np.float32)
    else:
        raise ValueError(f"Unknown chunk_repr: {mode}")


def featurize_obs(
    state: np.ndarray,
    mode: Literal["state", "pi0_encoder"],
    encoder_features: np.ndarray | None = None,
) -> np.ndarray:
    """state: (8,) or encoder_features: (~520,) → feature vector."""
    if mode == "state":
        return state.astype(np.float32)
    elif mode == "pi0_encoder":
        if encoder_features is None:
            raise ValueError(
                "obs_repr='pi0_encoder' requires encoder_features in chunks npz "
                "(run extract_encoder_features.py first)"
            )
        return encoder_features.astype(np.float32)
    else:
        raise ValueError(f"Unknown obs_repr: {mode}")


def label_y_wmt(terminal_state: dict | None) -> np.ndarray:
    """4-d label: (eef_x, eef_y, eef_z, gripper_open_avg)."""
    if terminal_state is None:
        return np.zeros(4, dtype=np.float32)
    eef = np.array(terminal_state["eef_pos"], dtype=np.float32)
    # gripper_qpos has 2 entries (left, right) — average for "open-ness"
    gripper_open = float(terminal_state["gripper_qpos"][0])
    return np.concatenate([eef, [gripper_open]]).astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ChunkDataset(Dataset):
    """Lazy-built in-memory dataset of (obs_feat, chunk_feat, y_v3mc, y_wmt, ...).

    Each chunk in each episode is one training example.
    """

    def __init__(
        self,
        chunk_dirs: list[Path | str],
        summary_index: dict,
        chunk_repr: Literal["pooled", "flattened"] = "flattened",
        obs_repr: Literal["state", "pi0_encoder"] = "state",
        gamma: float = 0.99,
        verbose: bool = True,
    ):
        self.examples: list[ChunkExample] = []
        self.chunk_repr = chunk_repr
        self.obs_repr = obs_repr
        self.gamma = gamma

        skipped_no_summary = 0
        skipped_no_features = 0
        total_files = 0

        for chunk_dir in chunk_dirs:
            chunk_dir = Path(chunk_dir)
            if not chunk_dir.exists():
                if verbose:
                    print(f"  warn: {chunk_dir} does not exist; skipping")
                continue
            for npz_path in sorted(chunk_dir.glob("eta_*.npz")):
                total_files += 1
                info = parse_npz_filename(npz_path)
                episode_id = info["episode_id"]

                if episode_id not in summary_index:
                    skipped_no_summary += 1
                    continue

                summary = summary_index[episode_id]
                y_wmt = label_y_wmt(summary["terminal_state"])
                success = summary["success"]

                try:
                    data = np.load(npz_path)
                except Exception as e:
                    if verbose:
                        print(f"  warn: could not load {npz_path}: {e}")
                    continue

                chunks = data["chunks"]
                states = data["states_at_chunk"]
                N = chunks.shape[0]
                if N == 0:
                    continue

                # Encoder features may be present after extract_encoder_features.py
                encoder_features = data.get("obs_features", None)
                if obs_repr == "pi0_encoder" and encoder_features is None:
                    skipped_no_features += 1
                    continue

                for t in range(N):
                    obs_feat = featurize_obs(
                        states[t], obs_repr,
                        encoder_features[t] if encoder_features is not None else None,
                    )
                    chunk_feat = featurize_chunk(chunks[t], chunk_repr)
                    # Discounted MC label: T = N (1-indexed last chunk), t is 0-indexed
                    y_v3mc = float(self.gamma ** (N - 1 - t)) * (1.0 if success else 0.0)
                    self.examples.append(ChunkExample(
                        obs_features=obs_feat,
                        chunk_features=chunk_feat,
                        y_v3mc=y_v3mc,
                        y_wmt=y_wmt,
                        task_id=info["task_id"],
                        episode_id=episode_id,
                        chunk_idx=t,
                        n_chunks=N,
                        eta=info["eta"],
                        success=success,
                    ))

        if verbose:
            print(
                f"  loaded {len(self.examples)} chunks from {total_files} npz files"
                f" (skipped {skipped_no_summary} no-summary, "
                f"{skipped_no_features} no-encoder-features)"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        return {
            "obs_features": torch.from_numpy(ex.obs_features),
            "chunk_features": torch.from_numpy(ex.chunk_features),
            "y_v3mc": torch.tensor(ex.y_v3mc, dtype=torch.float32),
            "y_wmt": torch.from_numpy(ex.y_wmt),
            "task_id": ex.task_id,
            "success": int(ex.success),
        }


# -----------------------------------------------------------------------------
# Episode-level split
# -----------------------------------------------------------------------------
def episode_split(
    dataset: ChunkDataset, val_frac: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Group chunks by episode and split episodes into train/val.

    Critical: this prevents chunk-level leakage where chunks from the same
    episode could end up in both train and val.
    """
    rng = np.random.default_rng(seed)
    episode_ids = sorted({ex.episode_id for ex in dataset.examples})
    perm = rng.permutation(len(episode_ids))
    shuffled = [episode_ids[i] for i in perm]
    n_val = int(len(shuffled) * val_frac)
    val_set = set(shuffled[:n_val])
    train_idx = [
        i for i, ex in enumerate(dataset.examples) if ex.episode_id not in val_set
    ]
    val_idx = [
        i for i, ex in enumerate(dataset.examples) if ex.episode_id in val_set
    ]
    return train_idx, val_idx
