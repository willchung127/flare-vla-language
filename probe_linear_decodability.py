"""probe_linear_decodability.py — linear probe of expert hidden states for prompt identity.

The attention probe showed expert L17 has cos≈0.993 between two prompts. That
means the *attention pattern* doesn't depend on prompt. But the residual stream
could still encode prompt info via MLP transformations.

This script answers: at each expert layer, is the prompt linearly decodable from
the residual stream? If YES even at L17, then activation steering at the residual
stream is viable. If NO by L17, info is genuinely lost and we need a different
intervention.

Methodology:
  1. For each init state in [0, n_inits):
       Run policy.infer with prompt_a → capture hidden state at each expert layer
       Run policy.infer with prompt_b → same
  2. For each layer: build (X: N x D, y: N) where N = 2 * n_inits, y in {0,1}
     Pool hidden state over action horizon (mean) → vector of length D
  3. Train 5-fold-CV logistic regression on each layer's (X, y)
  4. Plot CV AUC vs layer depth

Visualizations:
  probe_auc_per_layer.png        bar chart with CV mean ± std
  hidden_states_pca.png          PCA of hidden states colored by prompt, per layer
  probe_weight_norms.png         norm of probe weight vs layer (proxy for "how much signal")
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_probe_attn import (  # noqa: E402
    set_all_seeds,
    force_eager_attention,
)


# =============================================================================
# HIDDEN STATE CAPTURE — forward hooks on expert decoder layers
# =============================================================================

# Default regex matches openpi's PI0Pytorch path. Override via --expert-layer-pattern
# to support other architectures (LeRobot pi0.5, custom models, etc.).
DEFAULT_EXPERT_LAYER_PATTERN = r"^paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)$"
EXPERT_LAYER_RE = re.compile(DEFAULT_EXPERT_LAYER_PATTERN)


def list_likely_expert_layers(model, max_show: int = 60) -> list:
    """Debug helper — print all module paths that LOOK like decoder layers.

    Use this to discover the right --expert-layer-pattern for a new architecture.
    """
    import torch.nn as nn
    candidates = []
    for name, module in model.named_modules():
        nlow = name.lower()
        if "layer" in nlow and ("decode" in nlow or "expert" in nlow or
                                  "block" in nlow or
                                  (".layers." in name and name.count(".") >= 3)):
            # Look like decoder layers — has a layer index in the path
            candidates.append((name, type(module).__name__))
    print(f"[list-layers] Found {len(candidates)} candidate decoder-like modules:")
    for name, cls in candidates[:max_show]:
        print(f"  {name}  ({cls})")
    if len(candidates) > max_show:
        print(f"  ... and {len(candidates) - max_show} more")
    return candidates


class HiddenStateCapture:
    """Forward-hooks every module whose name matches `layer_regex`.

    For a decoder layer module, `output` is typically:
        (hidden_states, present_kv) for HF Gemma  — tuple, hidden_states at [0]
        or just hidden_states                     — tensor
    We extract the hidden_states tensor (shape (B, L, D)).

    Captured: dict layer_name -> list of (B, L, D) tensors, one per forward call.
    """

    def __init__(self, model: nn.Module, layer_regex: re.Pattern,
                  verbose: bool = True):
        self.model = model
        self.regex = layer_regex
        self.verbose = verbose
        self.captured: Dict[str, List[torch.Tensor]] = {}
        self._handles = []

    def _make_hook(self, name: str):
        captured = self.captured

        def hook(module, inputs, output):
            hs = None
            if isinstance(output, torch.Tensor):
                hs = output
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    hs = output[0]
            if hs is None or hs.dim() != 3:
                # Not a (B, L, D) hidden state — skip this call.
                captured.setdefault(name, []).append(None)
                return
            captured.setdefault(name, []).append(hs.detach().to("cpu"))

        return hook

    def __enter__(self):
        n_hooks = 0
        for name, module in self.model.named_modules():
            if self.regex.match(name):
                h = module.register_forward_hook(self._make_hook(name))
                self._handles.append(h)
                n_hooks += 1
        if n_hooks == 0:
            sample = [n for n, _ in self.model.named_modules()
                      if "gemma_expert" in n][:10]
            raise RuntimeError(
                f"No modules matched {self.regex.pattern}.\n"
                f"Sample gemma_expert paths seen: {sample}"
            )
        if self.verbose:
            print(f"[HiddenStateCapture] Hooked {n_hooks} modules matching "
                  f"{self.regex.pattern}")
        return self.captured

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
# POOLING + DATASET BUILD
# =============================================================================

def pool_hidden_state(hs: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Reduce a (B, L, D) hidden state to (D,) for the probe.

    For a single inference (B=1) and action horizon L:
        "mean" → average over horizon tokens (most stable)
        "last" → final token (per-token autoregressive convention)
        "max"  → element-wise max over horizon (rarely useful)

    Always casts to float32 (pi0 stores activations in BF16, but numpy doesn't
    support BF16 and downstream sklearn calls require float32).
    """
    if hs is None or hs.dim() != 3:
        raise ValueError(f"expected (B,L,D), got {None if hs is None else tuple(hs.shape)}")
    # Cast to float32 so .numpy() works and sklearn is happy.
    hs = hs.to(torch.float32)
    if hs.shape[0] != 1:
        # average over batch too
        hs = hs.mean(dim=0, keepdim=True)
    if mode == "mean":
        return hs[0].mean(dim=0)
    if mode == "last":
        return hs[0, -1]
    if mode == "max":
        return hs[0].max(dim=0).values
    raise ValueError(f"unknown pool mode: {mode}")


def build_dataset(captures_per_run: List[Tuple[int, Dict[str, List[torch.Tensor]]]],
                   pool_mode: str = "mean", call_idx: int = -1
                   ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int]]:
    """Combine per-run captures into per-layer feature matrices.

    captures_per_run: list of (label, captured_dict). label is 0 or 1 (prompt id).
    call_idx: which forward call to use (-1 = last). Each layer may be called
              multiple times during one inference (one per denoise step + maybe
              one for prefix).

    Returns:
      layer_X: {layer_name: (N, D) np.ndarray}
      y:       (N,) np.ndarray of labels
      run_ids: list of (which run each row came from) — for leave-one-run-out CV
    """
    all_layers = sorted({name for _, cap in captures_per_run for name in cap})
    layer_X: Dict[str, List[np.ndarray]] = {n: [] for n in all_layers}
    y_list: List[int] = []
    run_ids: List[int] = []

    for run_idx, (label, cap) in enumerate(captures_per_run):
        for name in all_layers:
            lst = cap.get(name, [])
            if not lst:
                # This layer was never hooked or never called for this run.
                # We can't build a consistent dataset — error out.
                raise RuntimeError(
                    f"run {run_idx} (label {label}) missing captures for layer {name}"
                )
            chosen = lst[call_idx] if abs(call_idx) <= len(lst) else None
            if chosen is None:
                raise RuntimeError(
                    f"run {run_idx} layer {name} call_idx={call_idx} returned None"
                )
            vec = pool_hidden_state(chosen, mode=pool_mode).numpy()
            layer_X[name].append(vec)
        y_list.append(label)
        run_ids.append(run_idx)

    return ({n: np.stack(v) for n, v in layer_X.items()},
            np.array(y_list),
            run_ids)


# =============================================================================
# LOGISTIC REGRESSION PROBE WITH CROSS-VALIDATION
# =============================================================================

def fit_probe_layer(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     seed: int = 0) -> Dict[str, float]:
    """Train L2-regularized logistic regression with k-fold CV; return AUC stats.

    With N small and D large (the typical case here), we MUST cross-validate;
    the training AUC will be ~1.0 even on random labels because the system is
    underdetermined. The CV AUC is what tells us if real signal exists.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        return {"auc_mean": float("nan"), "auc_std": 0.0,
                "n_folds": 0, "n_samples": int(len(y)),
                "reason": "only one class present"}
    n_splits = min(n_splits, int(min(np.bincount(y))))
    if n_splits < 2:
        return {"auc_mean": float("nan"), "auc_std": 0.0,
                "n_folds": 0, "n_samples": int(len(y)),
                "reason": f"not enough samples per class for {n_splits}-fold CV"}

    aucs = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        if len(np.unique(y[test_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[test_idx], prob))
    if not aucs:
        return {"auc_mean": float("nan"), "auc_std": 0.0,
                "n_folds": 0, "n_samples": int(len(y)),
                "reason": "all CV folds had single-class test sets"}

    # Also train on all data to get a weight-norm signal.
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)
    clf_full = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
    clf_full.fit(X_all, y)
    weight_norm = float(np.linalg.norm(clf_full.coef_))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_per_fold": [float(a) for a in aucs],
        "n_folds": len(aucs),
        "n_samples": int(len(y)),
        "weight_norm_all_data": weight_norm,
    }


# =============================================================================
# SANITY CHECKS
# =============================================================================

def _check_n_layers(captures: Dict[str, List]) -> Tuple[bool, str]:
    if not captures:
        return False, "no layers captured"
    # Expert decoder has 18 layers in Gemma — give some leeway.
    if not (10 <= len(captures) <= 40):
        return False, f"got {len(captures)} layers (expected ~18 expert layers)"
    return True, f"{len(captures)} expert layers"


def _check_captures_consistent(per_run: List[Tuple[int, Dict[str, List]]]) -> Tuple[bool, str]:
    """All runs should produce the same layer keys and (mostly) same shapes."""
    if not per_run:
        return False, "no runs"
    ref_keys = set(per_run[0][1].keys())
    for i, (_, cap) in enumerate(per_run[1:], start=1):
        if set(cap.keys()) != ref_keys:
            diff = ref_keys.symmetric_difference(cap.keys())
            return False, f"run {i}: layer keys differ; symmetric diff = {list(diff)[:5]}"
    return True, f"all {len(per_run)} runs have same layer keys ({len(ref_keys)})"


def _check_hidden_finite(per_run: List[Tuple[int, Dict[str, List]]]) -> Tuple[bool, str]:
    n_problems = 0
    for run_idx, (_, cap) in enumerate(per_run):
        for name, lst in cap.items():
            for i, h in enumerate(lst):
                if h is None:
                    continue
                if not torch.isfinite(h).all():
                    n_problems += 1
    if n_problems > 0:
        return False, f"{n_problems} non-finite hidden state tensors"
    return True, "all hidden states are finite"


def _check_probe_aucs_sane(aucs_per_layer: Dict[str, dict]) -> Tuple[bool, str]:
    """AUCs should be in [0, 1]; NaN-only would be suspicious."""
    aucs = [d["auc_mean"] for d in aucs_per_layer.values()
            if not np.isnan(d.get("auc_mean", float("nan")))]
    if not aucs:
        return False, "no non-NaN AUCs"
    if any(a < 0 or a > 1 for a in aucs):
        return False, f"AUC out of [0,1] range; min={min(aucs):.3f} max={max(aucs):.3f}"
    return True, f"{len(aucs)} AUCs in [{min(aucs):.3f}, {max(aucs):.3f}]"


def run_sanity_checks(per_run, aucs_per_layer) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if per_run:
        ok, msg = _check_n_layers(per_run[0][1])
        out["expert_layer_count"] = {"pass": ok, "message": msg}
    ok, msg = _check_captures_consistent(per_run)
    out["captures_consistent"] = {"pass": ok, "message": msg}
    ok, msg = _check_hidden_finite(per_run)
    out["hidden_states_finite"] = {"pass": ok, "message": msg}
    ok, msg = _check_probe_aucs_sane(aucs_per_layer)
    out["probe_aucs_sane"] = {"pass": ok, "message": msg}
    return out


# =============================================================================
# PLOTS
# =============================================================================

def make_plots(aucs_per_layer: Dict[str, dict],
                layer_X: Dict[str, np.ndarray],
                y: np.ndarray,
                out_dir: Path,
                title_suffix: str = "",
                layer_re: "re.Pattern" = None) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib unavailable — skipping plots")
        return []
    written: List[Path] = []

    # Sort layers by depth using the active layer regex (whose (\d+) group is depth).
    _re = layer_re if layer_re is not None else EXPERT_LAYER_RE
    def layer_idx(name: str) -> int:
        m = _re.match(name)
        return int(m.group(1)) if m else -1

    layers = sorted(aucs_per_layer.keys(), key=layer_idx)
    indices = [layer_idx(n) for n in layers]
    aucs = [aucs_per_layer[n].get("auc_mean", np.nan) for n in layers]
    stds = [aucs_per_layer[n].get("auc_std", 0.0) for n in layers]

    # ---------- Plot 1: AUC per layer ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(indices, aucs, yerr=stds, capsize=3, color="C0", alpha=0.8,
           edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="chance (0.5)")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("expert decoder layer depth")
    ax.set_ylabel("5-fold CV AUC (predict prompt identity)")
    ax.set_title(f"Linear decodability of prompt ID per expert layer{title_suffix}")
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(indices)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "probe_auc_per_layer.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # ---------- Plot 2: PCA of hidden states per layer ----------
    try:
        from sklearn.decomposition import PCA
        n_show = min(6, len(layers))
        # Show first, middle, last few layers
        if len(layers) <= n_show:
            show_layers = layers
        else:
            show_idx = np.linspace(0, len(layers) - 1, n_show).astype(int)
            show_layers = [layers[i] for i in show_idx]
        n_cols = 3
        n_rows = (len(show_layers) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = np.atleast_2d(axes).ravel()
        for i, name in enumerate(show_layers):
            ax = axes[i]
            X = layer_X[name]
            if X.shape[0] < 3:
                ax.text(0.5, 0.5, "too few samples", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
            for cls, color in [(0, "C0"), (1, "C1")]:
                mask = y == cls
                ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=f"prompt {cls}",
                           alpha=0.7, edgecolor="black", s=40)
            ax.set_title(f"L{layer_idx(name)} (CV AUC={aucs_per_layer[name]['auc_mean']:.3f})",
                         fontsize=10)
            ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)", fontsize=8)
            ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)", fontsize=8)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8, loc="best")
        for j in range(len(show_layers), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"PCA of expert hidden states by prompt{title_suffix}")
        fig.tight_layout()
        p = out_dir / "hidden_states_pca.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    except ImportError:
        print("[probe] sklearn.decomposition.PCA unavailable — skipping PCA plot")

    # ---------- Plot 3: weight norm per layer ----------
    norms = [aucs_per_layer[n].get("weight_norm_all_data", np.nan) for n in layers]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(indices, norms, "o-")
    ax.set_xlabel("expert decoder layer depth")
    ax.set_ylabel("L2 norm of full-data probe weight")
    ax.set_title(f"Probe weight magnitude per layer (proxy: signal strength){title_suffix}")
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "probe_weight_norms.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", default="pi0_libero")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--bddl-override", default=None)
    p.add_argument("--prompt-a", required=True)
    p.add_argument("--prompt-b", required=True)
    p.add_argument("--n-inits", type=int, default=25,
                   help="Number of distinct init states to use; total runs = 2*n_inits")
    p.add_argument("--call-idx", type=int, default=-1,
                   help="Which denoise step's hidden state to use (-1=last)")
    p.add_argument("--pool-mode", default="mean", choices=["mean", "last", "max"])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--expert-layer-pattern", default=DEFAULT_EXPERT_LAYER_PATTERN,
                   help="Regex matching expert decoder layer module paths. Default "
                        "is openpi PI0Pytorch convention; override for LeRobot, "
                        "converted models, etc. The (\\d+) group must capture the depth.")
    p.add_argument("--list-layers", action="store_true",
                   help="Load model and print all candidate decoder-layer paths, "
                        "then exit (useful for discovering --expert-layer-pattern for a new model)")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] importing openpi/libero ...", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa: E402
    from remote_multimode_matrix import (  # type: ignore  # noqa: E402
        format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION,
    )

    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.policy_config == "pi0_libero":
        ckpt_dir = str(Path.home() / "flare/checkpoints/pi0_libero_pt")
    else:
        from openpi.shared import download as openpi_download
        ckpt_dir = str(Path(openpi_download.maybe_download(
            f"gs://openpi-assets/checkpoints/{args.policy_config}")))
    print(f"[probe] checkpoint dir: {ckpt_dir}", flush=True)

    policy = policy_config.create_trained_policy(cfg, ckpt_dir)
    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            break
    if model is None:
        raise RuntimeError("Could not find pi0 PyTorch instance on policy")

    # --list-layers debug path: dump candidates and exit
    if args.list_layers:
        print(f"\n[probe] --list-layers: dumping candidate decoder layer paths in {type(model).__name__}")
        list_likely_expert_layers(model)
        sys.exit(0)

    # Compile the layer-matching regex (user-overridable for non-openpi architectures)
    expert_layer_re = re.compile(args.expert_layer_pattern)
    print(f"[probe] using expert layer pattern: {args.expert_layer_pattern}")

    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    model.flare_obs_state = None
    model.flare_eta_high = None
    model.flare_eta_low = None
    model.flare_noise_bias_direction = None
    model.flare_noise_bias_strength = 0.0
    model.eval()
    force_eager_attention(model, verbose=False)

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    if args.bddl_override:
        bddl = args.bddl_override
    else:
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(args.task_id)
    n_inits = min(args.n_inits, len(init_states))
    print(f"[probe] using {n_inits} init states (cap from {len(init_states)} available)")

    # Build per-run captures
    per_run: List[Tuple[int, Dict[str, List[torch.Tensor]]]] = []

    for init_i in range(n_inits):
        for prompt_label, prompt in [(0, args.prompt_a), (1, args.prompt_b)]:
            env = OffScreenRenderEnv(bddl_file_name=bddl,
                                      camera_heights=256, camera_widths=256)
            env.seed(args.env_seed + init_i)  # vary env seed too
            env.reset()
            obs = env.set_init_state(init_states[init_i])
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            obs_in = format_obs(obs, prompt)

            set_all_seeds(args.seed)
            with HiddenStateCapture(model, expert_layer_re, verbose=(init_i == 0 and prompt_label == 0)) as cap:
                _ = policy.infer(obs_in)
            per_run.append((prompt_label, dict(cap)))

            # Free env immediately to release EGL contexts.
            del env

            if init_i % 5 == 0 and prompt_label == 1:
                print(f"  init {init_i+1}/{n_inits} done (both prompts)", flush=True)

    print(f"[probe] collected {len(per_run)} runs total "
          f"({n_inits} inits × 2 prompts)")

    # Build dataset
    layer_X, y, run_ids = build_dataset(per_run, pool_mode=args.pool_mode,
                                          call_idx=args.call_idx)
    print(f"[probe] dataset: {len(layer_X)} layers, "
          f"each X has shape {next(iter(layer_X.values())).shape}, y has {len(y)} labels")
    print(f"[probe]   y class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Per-layer probe
    print(f"[probe] fitting per-layer {args.cv_folds}-fold CV probes ...")
    aucs_per_layer: Dict[str, dict] = {}
    for name in sorted(layer_X.keys()):
        result = fit_probe_layer(layer_X[name], y, n_splits=args.cv_folds, seed=args.seed)
        aucs_per_layer[name] = result

    # Sanity
    sanity = run_sanity_checks(per_run, aucs_per_layer)
    print("\n[probe] === SANITY CHECKS ===")
    for k, v in sanity.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {k}: {v['message']}")
    n_pass = sum(1 for v in sanity.values() if v["pass"])
    n_total = len(sanity)
    print(f"[probe] sanity: {n_pass}/{n_total} passing\n")

    # Print AUC table (sorted by depth, using the active layer regex)
    print("[probe] === PER-LAYER AUC ===")
    def layer_idx(n: str) -> int:
        m = expert_layer_re.match(n)
        return int(m.group(1)) if m else -1
    for name in sorted(aucs_per_layer.keys(), key=layer_idx):
        d = aucs_per_layer[name]
        idx = layer_idx(name)
        auc = d.get("auc_mean", float("nan"))
        std = d.get("auc_std", 0.0)
        wn = d.get("weight_norm_all_data", float("nan"))
        print(f"  L{idx:2d}: AUC = {auc:.3f} ± {std:.3f} | weight_norm = {wn:.2f}")

    # Save
    out_summary = {
        "policy_config": args.policy_config,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "n_inits": n_inits,
        "n_runs": len(per_run),
        "cv_folds": args.cv_folds,
        "pool_mode": args.pool_mode,
        "call_idx": args.call_idx,
        "aucs_per_layer": aucs_per_layer,
        "y_class_balance": {int(k): int(v) for k, v in
                            zip(*np.unique(y, return_counts=True))},
    }
    (out_dir / "probe_results.json").write_text(json.dumps(out_summary, indent=2))
    (out_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))
    # Save the raw layer_X + y for offline re-analysis
    np.savez_compressed(out_dir / "probe_data.npz",
                         y=y, **{f"X__{n}": layer_X[n] for n in layer_X})

    if args.plot:
        suffix = f"\nA: {args.prompt_a[:60]} | B: {args.prompt_b[:60]}"
        paths = make_plots(aucs_per_layer, layer_X, y, out_dir, suffix,
                            layer_re=expert_layer_re)
        for pp in paths:
            print(f"[probe] wrote {pp}")

    print(f"\n[probe] ✓ done. Files in {out_dir}")
    if n_pass < n_total:
        print(f"[probe] WARN: {n_total-n_pass} sanity check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
