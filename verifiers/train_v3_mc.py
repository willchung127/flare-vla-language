"""Train V3-MC: discounted Monte Carlo critic on (obs, chunk) → P(success).

Usage:
    python -m verifiers.train_v3_mc                    # default: state+flat, γ=0.99
    python -m verifiers.train_v3_mc --obs-repr pi0_encoder
    python -m verifiers.train_v3_mc --chunk-repr pooled
    python -m verifiers.train_v3_mc --gamma 0.95       # ablation

Pre-registered gate: held-out AUC ≥ 0.65.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
except ImportError:
    roc_auc_score = None
    brier_score_loss = None

from verifiers.data import ChunkDataset, episode_split, load_summary_jsons
from verifiers.models import V3MC, count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--day6a", default="/Users/william/flare/results/day6a_sde_sweep",
        help="Day 6a directory containing chunks/ subdir + eta_*.json summaries",
    )
    parser.add_argument(
        "--day7", default="/Users/william/flare/results/day7_outcome_scan",
        help="Day 7 directory containing chunks/ subdir + eta_*.json summaries",
    )
    parser.add_argument("--chunk-repr", default="flattened", choices=["pooled", "flattened"])
    parser.add_argument("--obs-repr", default="state", choices=["state", "pi0_encoder"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="/Users/william/flare/results/v3_mc_state.pt",
        help="Where to save the trained model checkpoint",
    )
    parser.add_argument(
        "--report",
        default="/Users/william/flare/results/v3_mc_report.json",
        help="JSON file with training metrics + gate verdict",
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
        chunk_repr=args.chunk_repr, obs_repr=args.obs_repr, gamma=args.gamma,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No training examples found. Check that chunks/ subdirs are populated "
            "and that eta_*.json summaries exist."
        )

    train_idx, val_idx = episode_split(dataset, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_idx)} chunks | Val: {len(val_idx)} chunks")
    print(
        f"  ({len({dataset.examples[i].episode_id for i in train_idx})} train episodes, "
        f"{len({dataset.examples[i].episode_id for i in val_idx})} val episodes)"
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # === Class weighting (positive labels are the rare class — failures) ===
    # In our data, success ~85%, so "positive (high y_v3mc)" is majority.
    # We want to upweight the rare class — failed episodes (y_v3mc=0).
    # pos_weight in BCE upweights the POSITIVE class. Since our positive class
    # is "succeeded" (the majority), we want pos_weight < 1.
    train_success_frac = (
        sum(dataset.examples[i].success for i in train_idx) / max(len(train_idx), 1)
    )
    pos_weight = (1.0 - train_success_frac) / max(train_success_frac, 0.01)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32)
    print(f"Train success frac: {train_success_frac:.3f} | pos_weight: {pos_weight:.3f}")

    # === Model ===
    obs_dim = dataset.examples[0].obs_features.shape[0]
    chunk_dim = dataset.examples[0].chunk_features.shape[0]
    model = V3MC(obs_dim, chunk_dim, hidden=args.hidden, dropout=args.dropout)
    print(f"V3MC: obs_dim={obs_dim}, chunk_dim={chunk_dim}, params={count_params(model):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # === Training loop ===
    best_val_auc = 0.0
    best_val_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            logits = model(batch["obs_features"], batch["chunk_features"])
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["y_v3mc"], pos_weight=pos_weight_t
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_logits, val_targets_v3mc, val_success = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                lg = model(batch["obs_features"], batch["chunk_features"])
                val_logits.append(lg.cpu())
                val_targets_v3mc.append(batch["y_v3mc"].cpu())
                val_success.append(batch["success"].cpu())
        val_logits = torch.cat(val_logits).numpy()
        val_targets_v3mc = torch.cat(val_targets_v3mc).numpy()
        val_success = torch.cat(val_success).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        # AUC: binary success-prediction task
        if roc_auc_score is not None and len(np.unique(val_success)) > 1:
            auc = float(roc_auc_score(val_success, val_probs))
            brier = float(brier_score_loss(val_success, val_probs))
        else:
            auc = float("nan")
            brier = float("nan")

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_auc": auc,
            "val_brier": brier,
        })

        if auc > best_val_auc:
            best_val_auc = auc
            best_val_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:3d}: train_loss={np.mean(train_losses):.4f} "
                f"val_auc={auc:.3f} val_brier={brier:.4f}"
            )

    # Save best model
    if best_val_state is None:
        best_val_state = model.state_dict()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_val_state,
        "config": vars(args),
        "obs_dim": obs_dim,
        "chunk_dim": chunk_dim,
        "obs_repr": args.obs_repr,
        "chunk_repr": args.chunk_repr,
        "best_val_auc": best_val_auc,
        "history": history,
        "gate_passed": best_val_auc >= 0.65,
    }, args.out)

    report = {
        "model": "V3-MC",
        "config": vars(args),
        "best_val_auc": best_val_auc,
        "final_val_brier": brier,
        "n_train_chunks": len(train_idx),
        "n_val_chunks": len(val_idx),
        "train_success_frac": train_success_frac,
        "pos_weight": pos_weight,
        "n_params": count_params(model),
        "gate_threshold": 0.65,
        "gate_passed": bool(best_val_auc >= 0.65),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {args.out}")
    print(f"Saved report: {args.report}")
    print(
        f"\nFINAL: best_val_AUC = {best_val_auc:.3f} "
        f"({'✓ GATE PASSED' if best_val_auc >= 0.65 else '✗ GATE FAILED'} — threshold 0.65)"
    )


if __name__ == "__main__":
    main()
