"""eval_lora_velocity.py — velocity probe comparison: LoRA vs baseline pi0_libero.

Runs probe_velocity_and_actions on the LoRA-trained model with paired prompts
that we've already characterized for baseline pi0_libero, then compares:

  baseline cos vs LoRA cos for the same prompt pairs.

If LoRA fixes the bypass, we expect:
  - cos(T0 prompt vs T1 prompt) drops further from noise floor (more language sensitivity)
  - cos(T0 prompt vs OOD rephrase) also drops (more sensitivity)

Reference baseline numbers (from earlier velocity calibration probes):

  NOISE_FLOOR (same prompt twice):           velocity cos = 1.0000
  T0_v1 (reorder bypass):                    velocity cos = 0.9997
  T0_vs_T2prompt (cross-task prompt):        velocity cos = 0.9915
  LANGUAGE_FOLLOWING (T0 vs T1):             velocity cos = 0.9762  ← key reference

After LoRA, we expect cos toward 0.95 or lower for the LANGUAGE_FOLLOWING pair
if the bypass is partially fixed.

================================================================================
USAGE
================================================================================

python3 eval_lora_velocity.py \\
    --checkpoint ~/flare/openpi/checkpoints/pi0_libero_cf_lora/variant_c_v1/2999 \\
    --config-name pi0_libero_cf_lora \\
    --variant-label variant_c_natural_cf \\
    --out-dir ~/flare/results/lora_ablation/variant_c/velocity_probes \\
    --baseline-velocity-dirs ~/flare/results/probe_velocity_T0_LANGUAGE_FOLLOWING \\
                              ~/flare/results/probe_velocity_T0_NOISE_FLOOR \\
                              ~/flare/results/probe_velocity_T0_vs_T2prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


PAIRS = [
    ("LANGUAGE_FOLLOWING",
     "put both the alphabet soup and the tomato sauce in the basket",
     "put both the cream cheese box and the butter in the basket",
     "Both trained prompts; baseline ~0.9762; key reference"),
    ("T0_vs_T0_tomato_only",
     "put both the alphabet soup and the tomato sauce in the basket",
     "pick up the tomato sauce and put it in the basket",
     "Original T0 prompt vs OOD-shaped single-target tomato prompt"),
    ("T0_vs_T2prompt",
     "put both the alphabet soup and the tomato sauce in the basket",
     "turn on the stove and put the moka pot on it",
     "Original T0 prompt vs unrelated kitchen prompt; baseline ~0.9915"),
]


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    n = (af.norm() * bf.norm()).clamp(min=eps)
    return float((af * bf).sum() / n)


def load_baseline_cos(baseline_dirs: List[Path]) -> Dict[str, dict]:
    """Load previously-computed velocity cos from baseline probe runs."""
    baseline = {}
    for d in baseline_dirs:
        d = Path(d).expanduser()
        log_candidates = [
            d.parent / f"{d.name}.log",
            d / "divergence.json",
            d / "summary.json",
        ]
        # First try parsing the log file
        for cand in log_candidates:
            if cand.exists() and cand.suffix == ".log":
                txt = cand.read_text()
                # Look for "velocity cos (mean over steps): X.XXXX"
                import re
                m_vel = re.search(r"velocity cos \(mean over steps\):\s*([\d.]+)", txt)
                m_act = re.search(r"action chunk cos \(overall\):\s*([\d.]+)", txt)
                if m_vel:
                    baseline[d.name] = {
                        "velocity_cos_mean": float(m_vel.group(1)),
                        "action_chunk_cos": float(m_act.group(1)) if m_act else None,
                        "source": str(cand),
                    }
                    break
            elif cand.exists() and cand.suffix == ".json":
                try:
                    j = json.loads(cand.read_text())
                    baseline[d.name] = {
                        "velocity_cos_mean": j.get("velocity_cos_mean") or j.get("cos_velocity"),
                        "action_chunk_cos": j.get("action_chunk_cos") or j.get("cos_action"),
                        "source": str(cand),
                    }
                    break
                except Exception:
                    pass
    return baseline


def setup_policy_and_env(config_name: str, checkpoint_dir: Path,
                          task_suite: str, task_id: int):
    """Load LoRA policy + LIBERO env. Shared across pair runs."""
    print(f"[load] policy: config={config_name!r}, ckpt={checkpoint_dir}", flush=True)
    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa
    Pi0 = apply_patch()

    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import remote_multimode_matrix as mmm  # noqa
    from remote_multimode_matrix import format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION

    cfg = _c.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, str(checkpoint_dir))

    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            break

    # Disable any sde/guidance attrs
    if model is not None:
        for a in ("flare_eta", "flare_alpha", "flare_noise_bias_strength"):
            if hasattr(model, a):
                setattr(model, a, 0.0)
        for a in ("flare_verifier_fn", "flare_obs_state", "flare_eta_high",
                  "flare_eta_low", "flare_noise_bias_direction"):
            if hasattr(model, a):
                setattr(model, a, None)
        if hasattr(model, "eval"):
            model.eval()

    bench = benchmark.get_benchmark_dict()[task_suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                              camera_heights=256, camera_widths=256)

    return {
        "policy": policy, "model": model,
        "env": env, "init_states": init_states,
        "format_obs": format_obs, "NUM_STEPS_WAIT": NUM_STEPS_WAIT,
        "LIBERO_DUMMY_ACTION": LIBERO_DUMMY_ACTION,
    }


def reset_to_init(env, init_state, format_obs_fn, dummy_action,
                   num_wait: int, seed: int = 7000):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(num_wait):
        obs, _, _, _ = env.step(dummy_action)
    return obs


def capture_velocities_for_prompt(policy, model, env_pack, init_state, prompt: str):
    """Run policy.infer() with a velocity capture hook; return list of (1,H,D) tensors."""
    captured = []

    # Use a forward-pre-hook on denoise_step (matching probe_velocity infrastructure)
    if model is None:
        # JAX-only model; we'd need a different mechanism
        raise NotImplementedError(
            "Velocity capture currently requires a PyTorch model attribute. "
            "openpi's JAX policy doesn't expose denoise_step as a Python module. "
            "For JAX-only models, modify pi0.py to sow velocities and read intermediates."
        )

    # Save original denoise_step
    orig_denoise = type(model).denoise_step

    def wrapped(self, *args, **kwargs):
        v = orig_denoise(self, *args, **kwargs)
        captured.append(v.detach().cpu())
        return v

    type(model).denoise_step = wrapped
    try:
        obs = reset_to_init(
            env_pack["env"], init_state,
            env_pack["format_obs"], env_pack["LIBERO_DUMMY_ACTION"],
            env_pack["NUM_STEPS_WAIT"],
        )
        obs_in = env_pack["format_obs"](obs, prompt)
        _ = policy.infer(obs_in)
    finally:
        type(model).denoise_step = orig_denoise

    return captured


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", default="pi0_libero_cf_lora")
    p.add_argument("--variant-label", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline-velocity-dirs", nargs="*", default=[],
                   help="Existing baseline probe_velocity directories or .log files")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-idx", type=int, default=0,
                   help="Init state index for the probe (matches earlier probes)")
    p.add_argument("--real-action-dim", type=int, default=7)
    p.add_argument("--wandb-run-id", default=None,
                   help="If set, log velocity metrics to this wandb run")
    p.add_argument("--wandb-project", default="flare_cf_lora")
    p.add_argument("--training-step", type=int, default=None,
                   help="If set, log velocity metrics at this wandb step")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== VELOCITY EVAL: {args.variant_label} ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  pairs: {len(PAIRS)}")
    print()

    # Load baseline cosines for reference
    baseline = load_baseline_cos([Path(d) for d in args.baseline_velocity_dirs])
    if baseline:
        print(f"[baseline] loaded {len(baseline)} reference probe(s)")
        for name, d in baseline.items():
            print(f"  {name}: vel_cos={d.get('velocity_cos_mean'):.4f}, "
                  f"act_cos={d.get('action_chunk_cos')}")
    else:
        print(f"[warn] no baseline references loaded")

    # Setup policy + env
    env_pack = setup_policy_and_env(
        args.config_name,
        Path(args.checkpoint).expanduser().resolve(),
        args.task_suite, args.task_id,
    )
    init_state = env_pack["init_states"][args.init_idx]

    # Run each pair
    R = args.real_action_dim
    results = {}
    for label, prompt_a, prompt_b, desc in PAIRS:
        print(f"\n[probe] === {label} ===")
        print(f"  prompt_a: {prompt_a!r}")
        print(f"  prompt_b: {prompt_b!r}")
        try:
            t0 = time.time()
            va = capture_velocities_for_prompt(
                env_pack["policy"], env_pack["model"],
                env_pack, init_state, prompt_a,
            )
            vb = capture_velocities_for_prompt(
                env_pack["policy"], env_pack["model"],
                env_pack, init_state, prompt_b,
            )
            elapsed = time.time() - t0
            print(f"  captured {len(va)} velocities for A, {len(vb)} for B "
                  f"in {elapsed:.1f}s")

            # Compute per-step cos
            cos_all = [cos_sim(a, b) for a, b in zip(va, vb)]
            cos_real = [cos_sim(a[..., :R], b[..., :R]) for a, b in zip(va, vb)]
            cos_exec = [cos_sim(a[..., :5, :R], b[..., :5, :R]) for a, b in zip(va, vb)]

            results[label] = {
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "description": desc,
                "n_denoise_steps": len(va),
                "cos_all_32dims": float(np.mean(cos_all)),
                "cos_real_7dims": float(np.mean(cos_real)),
                "cos_executed_positions_0_4": float(np.mean(cos_exec)),
                "cos_per_step_all": cos_all,
                "cos_per_step_real7": cos_real,
            }
            print(f"  cos(all)={np.mean(cos_all):.4f}  "
                  f"cos(real7)={np.mean(cos_real):.4f}  "
                  f"cos(exec[0:5])={np.mean(cos_exec):.4f}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[label] = {"error": str(e)}

    # Build comparison table
    print(f"\n=== COMPARISON TABLE ===")
    print(f"  {'pair':<25s} {'baseline':>10s} {'LoRA':>10s} {'delta':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    comparison = {}
    for label in [p[0] for p in PAIRS]:
        if label not in results or "error" in results[label]:
            continue
        lora_cos = results[label]["cos_real_7dims"]

        # Try to match to baseline
        bl_cos = None
        baseline_match = None
        for bname, bdata in baseline.items():
            if label.lower() in bname.lower() or any(t in bname.lower() for t in label.lower().split("_")):
                bl_cos = bdata.get("velocity_cos_mean")
                baseline_match = bname
                break

        if bl_cos is not None:
            delta = lora_cos - bl_cos
            arrow = "↓" if delta < -0.005 else ("↑" if delta > 0.005 else "≈")
            print(f"  {label:<25s} {bl_cos:>10.4f} {lora_cos:>10.4f} {delta:>+9.4f} {arrow}")
            comparison[label] = {
                "baseline_match": baseline_match,
                "baseline_cos": bl_cos,
                "lora_cos": lora_cos,
                "delta": delta,
            }
        else:
            print(f"  {label:<25s} {'?':>10s} {lora_cos:>10.4f}  (no baseline match)")
            comparison[label] = {"baseline_match": None, "lora_cos": lora_cos}

    # Save all data
    out_data = {
        "variant_label": args.variant_label,
        "checkpoint": str(args.checkpoint),
        "baseline_references": baseline,
        "lora_results": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ["cos_per_step_all", "cos_per_step_real7"]}
                          for k, v in results.items()},
        "comparison": comparison,
    }
    (out_dir / "velocity_eval.json").write_text(json.dumps(out_data, indent=2, default=str))

    # Make comparison plot
    try:
        make_velocity_plot(comparison, args.variant_label, out_dir)
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # Optional wandb logging
    if args.wandb_run_id is not None:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_id,
                resume="allow",
            )
            log_payload = {}
            for label, res in results.items():
                if "error" in res:
                    continue
                log_payload[f"velocity/{label}/cos_all_32dims"] = res["cos_all_32dims"]
                log_payload[f"velocity/{label}/cos_real_7dims"] = res["cos_real_7dims"]
                log_payload[f"velocity/{label}/cos_executed_pos_0_4"] = res["cos_executed_positions_0_4"]
            for label, comp in comparison.items():
                if "baseline_cos" in comp:
                    log_payload[f"velocity/{label}/baseline_cos"] = comp["baseline_cos"]
                    log_payload[f"velocity/{label}/delta_from_baseline"] = comp["delta"]

            if args.training_step is not None:
                wandb.log(log_payload, step=args.training_step)
            else:
                wandb.log(log_payload)

            plot_path = out_dir / "velocity_cos_comparison.png"
            if plot_path.exists():
                wandb.log({"velocity/plot/cos_comparison": wandb.Image(str(plot_path))})

            print(f"\n[velocity] logged metrics to wandb run {args.wandb_run_id}")
        except Exception as e:
            print(f"\n[warn] wandb logging failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] saved to {out_dir}")


def make_velocity_plot(comparison: Dict, variant_label: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in comparison.items() if "baseline_cos" in v}
    if not valid:
        print(f"[warn] no baseline comparisons; skipping plot")
        return

    labels = list(valid.keys())
    bl_vals = [valid[l]["baseline_cos"] for l in labels]
    lora_vals = [valid[l]["lora_cos"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, bl_vals, w, label="Baseline pi0_libero", color="steelblue")
    ax.bar(x + w/2, lora_vals, w, label=f"LoRA ({variant_label})", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Velocity cos (real 7 dims, mean over steps)")
    ax.set_ylim(0.9, 1.005)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="Noise floor")
    ax.set_title(f"Velocity cos comparison — {variant_label}")
    ax.legend()
    for i, (b, l) in enumerate(zip(bl_vals, lora_vals)):
        ax.text(i - w/2, b + 0.001, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, l + 0.001, f"{l:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_cos_comparison.png", dpi=120)
    plt.close(fig)
    print(f"  [plot] wrote velocity_cos_comparison.png")


if __name__ == "__main__":
    main()
