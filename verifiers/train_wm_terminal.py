"""Train WM-terminal: nano outcome predictor on (obs, chunk) → terminal eef+gripper.

Usage:
    python -m verifiers.train_wm_terminal
    python -m verifiers.train_wm_terminal --obs-repr pi0_encoder
    python -m verifiers.train_wm_terminal --chunk-repr pooled

Also computes per-task success centroid (`task_targets`) for inference scoring.

Pre-registered gate: held-out MSE ≤ 1.5 × train MSE (overfitting check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from verifiers.data import ChunkDataset, episode_split, load_summary_jsons
from verifiers.models import WMTerminal, count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--day6a", default="/Users/william/flare/results/day6a_sde_sweep",
    )
    parser.add_argument(
        "--day7", default="/Users/william/flare/results/day7_outcome_scan",
    )
    parser.add_argument("--chunk-repr", default="flattened", choices=["pooled", "flattened"])
    parser.add_argument("--obs-repr", default="state", choices=["state", "pi0_encoder"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="/Users/william/flare/results/wm_terminal_state.pt"
    )
    parser.add_argument(
        "--report", default="/Users/william/flare/results/wm_terminal_report.json"
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # === Load data ===
    print("Loading summaries...")
    summary_index = load_summary_jsons(args.day6a, args.day7)
    print(f"  {len(summary_index)} episodes with terminal_state recorded")

    chunk_dirs = []
    for d in [args.day6a, args.day7]:
        chunks_subdir = Path(d) / "chunks"
        if chunks_subdir.exists():
            chunk_dirs.append(chunks_subdir)
    print(f"Loading chunks from: {chunk_dirs}")

    dataset = ChunkDataset(
        chunk_dirs, summary_index,
        chunk_repr=args.chunk_repr, obs_repr=args.obs_repr,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training examples found.")

    train_idx, val_idx = episode_split(dataset, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_idx)} chunks | Val: {len(val_idx)} chunks")

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
    )

    # === Per-dim label normalization (so eef [meters] and gripper [0-0.04] are commensurate) ===
    train_y = np.stack([dataset.examples[i].y_wmt for i in train_idx])  # (N, 4)
    label_mean = train_y.mean(axis=0).astype(np.float32)
    label_std = (train_y.std(axis=0) + 1e-6).astype(np.float32)
    print(f"Label mean: {label_mean.tolist()}")
    print(f"Label std:  {label_std.tolist()}")

    label_mean_t = torch.from_numpy(label_mean)
    label_std_t = torch.from_numpy(label_std)

    # === Per-task success centroid (the "target" for inference scoring) ===
    task_targets = {}
    for task_id in range(10):
        successes_for_task = [
            dataset.examples[i].y_wmt
            for i in train_idx
            if dataset.examples[i].task_id == task_id
            and dataset.examples[i].success
        ]
        if len(successes_for_task) > 0:
            task_targets[task_id] = np.stack(successes_for_task).mean(axis=0).astype(np.float32)
        else:
            task_targets[task_id] = label_mean.copy()
    print("Per-task success centroids:")
    for task_id, tgt in task_targets.items():
        print(f"  task {task_id}: {tgt.tolist()}")

    # === Model ===
    obs_dim = dataset.examples[0].obs_features.shape[0]
    chunk_dim = dataset.examples[0].chunk_features.shape[0]
    model = WMTerminal(obs_dim, chunk_dim, hidden=args.hidden, dropout=args.dropout)
    print(
        f"WMTerminal: obs_dim={obs_dim}, chunk_dim={chunk_dim}, "
        f"params={count_params(model):,}"
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # === Training loop ===
    best_val_mse = float("inf")
    best_val_state = None
    history = []
    train_mse_final = float("nan")
    val_mse_final = float("nan")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            pred = model(batch["obs_features"], batch["chunk_features"])
            normalized_pred = (pred - label_mean_t) / label_std_t
            normalized_tgt = (batch["y_wmt"] - label_mean_t) / label_std_t
            loss = F.mse_loss(normalized_pred, normalized_tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        per_dim_errors = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["obs_features"], batch["chunk_features"])
                normalized_pred = (pred - label_mean_t) / label_std_t
                normalized_tgt = (batch["y_wmt"] - label_mean_t) / label_std_t
                loss = F.mse_loss(normalized_pred, normalized_tgt)
                val_losses.append(loss.item())
                per_dim_errors.append(
                    (pred - batch["y_wmt"]).abs().mean(dim=0).cpu().numpy()
                )
        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))
        per_dim_mae = np.stack(per_dim_errors).mean(axis=0)  # (4,)
        train_mse_final = train_mse
        val_mse_final = val_mse

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_per_dim_mae": per_dim_mae.tolist(),
            "val_to_train_ratio": val_mse / train_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:3d}: train_mse={train_mse:.4f} val_mse={val_mse:.4f} "
                f"val/train={val_mse/train_mse:.2f} | per-dim MAE: "
                f"eef_x={per_dim_mae[0]:.3f} eef_y={per_dim_mae[1]:.3f} "
                f"eef_z={per_dim_mae[2]:.3f} grip={per_dim_mae[3]:.3f}"
            )

    if best_val_state is None:
        best_val_state = model.state_dict()

    overfitting_ratio = val_mse_final / train_mse_final if train_mse_final > 0 else float("inf")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_val_state,
        "config": vars(args),
        "obs_dim": obs_dim,
        "chunk_dim": chunk_dim,
        "obs_repr": args.obs_repr,
        "chunk_repr": args.chunk_repr,
        "label_mean": label_mean,
        "label_std": label_std,
        "task_targets": {int(k): v.tolist() for k, v in task_targets.items()},
        "best_val_mse": best_val_mse,
        "train_mse_final": train_mse_final,
        "val_mse_final": val_mse_final,
        "overfitting_ratio": overfitting_ratio,
        "history": history,
        "gate_passed": overfitting_ratio <= 1.5,
    }, args.out)

    report = {
        "model": "WM-terminal",
        "config": vars(args),
        "best_val_mse": best_val_mse,
        "train_mse_final": train_mse_final,
        "val_mse_final": val_mse_final,
        "overfitting_ratio": overfitting_ratio,
        "n_train_chunks": len(train_idx),
        "n_val_chunks": len(val_idx),
        "n_params": count_params(model),
        "per_task_target_counts": {
            int(t): sum(
                1 for i in train_idx
                if dataset.examples[i].task_id == t and dataset.examples[i].success
            )
            for t in range(10)
        },
        "gate_threshold": 1.5,
        "gate_passed": bool(overfitting_ratio <= 1.5),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {args.out}")
    print(f"Saved report: {args.report}")
    print(
        f"\nFINAL: train_mse={train_mse_final:.4f}, val_mse={val_mse_final:.4f}, "
        f"val/train={overfitting_ratio:.2f} "
        f"({'✓ GATE PASSED' if overfitting_ratio <= 1.5 else '✗ GATE FAILED'} — threshold 1.5)"
    )


if __name__ == "__main__":
    main()
