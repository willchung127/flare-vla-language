"""Build the V2_outcome kNN cache from Day 6a + Day 7 successful chunks.

The cache stores (obs_features ⊕ pooled_chunk_features, terminal_eef_pos) tuples
from CLOSED-LOOP rollouts that succeeded. V2_outcome scorer queries this cache
at inference: for each candidate chunk, find k nearest neighbors and score by
-mean distance.

Output: ~/flare/results/v2_outcome_cache.npz with keys:
    features: (N, D_obs + 14)         joint (obs, pooled chunk) features
    targets:  (N, 3)                  terminal eef positions
    task_ids: (N,)                    task IDs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from verifiers.data import (
    ChunkDataset, episode_split, featurize_chunk, load_summary_jsons,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6a", default="/Users/william/flare/results/day6a_sde_sweep")
    parser.add_argument("--day7", default="/Users/william/flare/results/day7_outcome_scan")
    parser.add_argument("--obs-repr", default="state", choices=["state", "pi0_encoder"])
    parser.add_argument(
        "--out", default="/Users/william/flare/results/v2_outcome_cache.npz"
    )
    args = parser.parse_args()

    print("Loading summaries...")
    summary_index = load_summary_jsons(args.day6a, args.day7)

    chunk_dirs = []
    for d in [args.day6a, args.day7]:
        chunks_subdir = Path(d) / "chunks"
        if chunks_subdir.exists():
            chunk_dirs.append(chunks_subdir)

    # ChunkDataset already builds (obs_feat, chunk_feat, ..., success) per chunk.
    # For V2, we use POOLED chunk features (compact, faster kNN) and keep only
    # successful examples.
    dataset = ChunkDataset(
        chunk_dirs, summary_index,
        chunk_repr="pooled", obs_repr=args.obs_repr,
    )

    features = []
    targets = []
    task_ids = []
    for ex in dataset.examples:
        if not ex.success:
            continue
        joint_feat = np.concatenate([ex.obs_features, ex.chunk_features])
        features.append(joint_feat)
        # Terminal eef (first 3 of y_wmt)
        targets.append(ex.y_wmt[:3])
        task_ids.append(ex.task_id)

    features = np.stack(features).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    task_ids = np.array(task_ids, dtype=np.int32)

    print(f"Cache: {features.shape[0]} entries, {features.shape[1]}-d features")
    print(f"  per-task counts:")
    for t in range(10):
        n_t = int((task_ids == t).sum())
        print(f"    task {t}: {n_t}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        targets=targets,
        task_ids=task_ids,
    )
    print(f"\nSaved {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
