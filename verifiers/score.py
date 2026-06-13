"""Inference-time scoring with all four verifiers.

Each scorer takes:
    obs_features: (D_obs,) or (1, D_obs)        — observation features for the current step
    candidate_chunks: (K, 50, 7)                — K candidate chunks from π₀
    task_id: int (only used by WM-terminal)

And returns:
    scores: (K,) np.ndarray                      — higher = better; argmax is selected
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from verifiers.data import featurize_chunk
from verifiers.models import V3MC, WMTerminal


# =============================================================================
# V1 — Chunk-medoid (trivial baseline, zero training)
# =============================================================================
def _dtw_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance between two (H, D) chunks."""
    H = a.shape[0]
    cost = np.linalg.norm(a[:, None] - b[None], axis=-1)
    D = np.full((H + 1, H + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, H + 1):
        for j in range(1, H + 1):
            D[i, j] = cost[i - 1, j - 1] + min(
                D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
            )
    return float(D[H, H])


def score_v1_medoid(candidate_chunks: np.ndarray) -> np.ndarray:
    """V1: pick the most central chunk (minimizes mean DTW to others)."""
    K = candidate_chunks.shape[0]
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = _dtw_pairwise(candidate_chunks[i], candidate_chunks[j])
    # Score = -mean distance to others (higher = more central)
    return -D.sum(axis=1) / max(K - 1, 1)


# =============================================================================
# V2_outcome — kNN on success-terminal-eef cache
# =============================================================================
class V2OutcomeScorer:
    """kNN scorer on (obs_features ⊕ pooled_chunk_features) against a cache of
    features from SUCCESSFUL closed-loop chunks. Higher score = closer to a
    known-success neighborhood.
    """

    def __init__(self, cache_path: str, k: int = 5):
        cache = np.load(cache_path)
        self.features = cache["features"]  # (N_cache, D_feat)
        self.targets = cache.get("targets", None)  # (N_cache, 3) terminal eef (optional)
        self.k = k
        self.feat_norm = np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-6

    def score(
        self, obs_features: np.ndarray, candidate_chunks: np.ndarray
    ) -> np.ndarray:
        """obs_features: (D_obs,), candidate_chunks: (K, 50, 7)."""
        K = candidate_chunks.shape[0]
        # Build query features: (K, D_obs + 14)
        pooled = np.stack([
            featurize_chunk(c, "pooled") for c in candidate_chunks
        ])  # (K, 14)
        queries = np.concatenate([
            np.tile(obs_features, (K, 1)),  # (K, D_obs)
            pooled,
        ], axis=1)  # (K, D_obs + 14)

        # Compute distances
        # NB: features and queries should be commensurate; assume cache was built
        # with the same obs_repr + same chunk pooling.
        dists = np.linalg.norm(
            queries[:, None] - self.features[None], axis=-1
        )  # (K, N_cache)

        # Top-k mean distance (lower distance = higher score)
        kk = min(self.k, dists.shape[1])
        topk_dists = np.partition(dists, kk - 1, axis=1)[:, :kk].mean(axis=1)
        return -topk_dists


# =============================================================================
# V3-MC — Discounted Monte Carlo critic
# =============================================================================
class V3MCScorer:
    """Loads a trained V3-MC model and scores candidate chunks."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = V3MC(
            obs_dim=ckpt["obs_dim"],
            chunk_dim=ckpt["chunk_dim"],
            hidden=ckpt["config"]["hidden"],
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.chunk_repr = ckpt["chunk_repr"]
        self.obs_repr = ckpt["obs_repr"]

    def score(
        self, obs_features: np.ndarray, candidate_chunks: np.ndarray
    ) -> np.ndarray:
        K = candidate_chunks.shape[0]
        chunk_feats = np.stack(
            [featurize_chunk(c, self.chunk_repr) for c in candidate_chunks]
        )
        obs_rep = np.tile(obs_features, (K, 1))
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_rep).float().to(self.device)
            chunk_t = torch.from_numpy(chunk_feats).float().to(self.device)
            scores = self.model.score(obs_t, chunk_t).cpu().numpy()
        return scores


# =============================================================================
# WM-terminal — Nano outcome predictor
# =============================================================================
class WMTerminalScorer:
    """Loads a trained WM-terminal model and scores by -L2 to per-task target."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WMTerminal(
            obs_dim=ckpt["obs_dim"],
            chunk_dim=ckpt["chunk_dim"],
            hidden=ckpt["config"]["hidden"],
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.chunk_repr = ckpt["chunk_repr"]
        self.obs_repr = ckpt["obs_repr"]
        self.task_targets = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in ckpt["task_targets"].items()
        }
        self.label_std = ckpt["label_std"]  # (4,) — for Mahalanobis-style distance

    def score(
        self,
        obs_features: np.ndarray,
        candidate_chunks: np.ndarray,
        task_id: int,
    ) -> np.ndarray:
        K = candidate_chunks.shape[0]
        chunk_feats = np.stack(
            [featurize_chunk(c, self.chunk_repr) for c in candidate_chunks]
        )
        obs_rep = np.tile(obs_features, (K, 1))
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_rep).float().to(self.device)
            chunk_t = torch.from_numpy(chunk_feats).float().to(self.device)
            pred = self.model(obs_t, chunk_t).cpu().numpy()  # (K, 4)
        target = self.task_targets[int(task_id)]  # (4,)
        # Normalize by per-dim std so eef and gripper are commensurate
        scaled_diff = (pred - target) / self.label_std
        scores = -np.linalg.norm(scaled_diff, axis=-1)  # (K,)
        return scores


# =============================================================================
# Composite scorer factory for BoN runner
# =============================================================================
def build_scorers(
    v3mc_ckpt: Optional[str] = None,
    wmt_ckpt: Optional[str] = None,
    v2_cache: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """Build a dict of available scorer functions for use in BoN runner."""
    scorers = {"V1": lambda obs, chunks, task_id=None: score_v1_medoid(chunks)}
    if v2_cache is not None and Path(v2_cache).exists():
        v2 = V2OutcomeScorer(v2_cache)
        scorers["V2_outcome"] = lambda obs, chunks, task_id=None: v2.score(obs, chunks)
    if v3mc_ckpt is not None and Path(v3mc_ckpt).exists():
        v3 = V3MCScorer(v3mc_ckpt, device=device)
        scorers["V3_MC"] = lambda obs, chunks, task_id=None: v3.score(obs, chunks)
    if wmt_ckpt is not None and Path(wmt_ckpt).exists():
        wmt = WMTerminalScorer(wmt_ckpt, device=device)
        scorers["WM_terminal"] = lambda obs, chunks, task_id: wmt.score(
            obs, chunks, task_id
        )
    return scorers
