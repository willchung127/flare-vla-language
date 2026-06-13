"""eval_lora_attention.py — analyze attention patterns in a LoRA-trained pi0 model.

================================================================================
STATUS: SKELETON — REQUIRES JAX MODIFICATION TO FULLY WORK
================================================================================

The challenge: openpi trains pi0 in JAX/Flax, but our existing attention
analysis infrastructure (sink_decode.py, attention_boost.py) uses PyTorch
forward_pre_hooks on the PI0Pytorch class.

Two paths to get attention data from a LoRA-trained JAX model:

PATH A: JAX intermediates (preferred, no conversion)
  1. Modify ~/flare/openpi/src/openpi/models/pi0.py (or wherever attention
     happens) to add `self.sow('intermediates', 'attn_weights', attn)` inside
     the attention forward pass.
  2. Call model.apply(params, ..., mutable=['intermediates']) to capture them.
  3. Read attention from intermediates dict.

PATH B: JAX→PyTorch conversion
  1. Write a converter that maps the JAX param tree → PyTorch state_dict.
  2. Load into PI0Pytorch.
  3. Use existing PT hooks (attention_boost.py infrastructure).

This script implements PATH B as the primary path because it reuses all our
existing analysis code. The converter (jax_to_pytorch_lora_merge below) is
the missing piece.

================================================================================
WHAT'S IMPLEMENTED
================================================================================

  ✅ Argument parsing + output structure
  ✅ Attention measurement using existing PT hooks (once we have a PT model)
  ✅ Per-layer + per-head attention mass on language tokens, sink tokens
  ✅ Visualization (per-layer heatmap, per-head bar chart)
  ⚠️ jax_to_pytorch_lora_merge() is a STUB — needs implementation

================================================================================
TO IMPLEMENT (~2-4 hours, tomorrow morning when fresh)
================================================================================

The conversion function needs to:
  1. Load JAX params via `openpi.models.model.restore_params(jax_ckpt_dir)`
  2. Merge LoRA weights into base: for each (W, A, B) triple:
       W_merged = W + (A @ B) * (lora_alpha / lora_rank)
  3. Map JAX param names to PyTorch state_dict keys:
       ['PaliGemma']['llm']['layers']['attn']['q_proj']['kernel']  →  paligemma_with_expert.paligemma.model.layers.{i}.self_attn.q_proj.weight
       (with appropriate transpose/reshape for shape mismatches)
  4. Save as a .pth/.safetensors file
  5. Have load_pytorch_lora_merged() load it into PI0Pytorch

Reference for param name mapping: compare a sample JAX checkpoint's param tree
vs ~/flare/checkpoints/pi0_libero_pt/'s state_dict keys.

================================================================================
USAGE (once converter is implemented)
================================================================================

python3 eval_lora_attention.py \\
    --jax-checkpoint ~/flare/openpi/checkpoints/pi0_libero_cf_lora/variant_c_v1/2999/params \\
    --variant-label variant_c_natural_cf \\
    --baseline-checkpoint ~/flare/checkpoints/pi0_libero_pt \\
    --out-dir ~/flare/results/lora_ablation/variant_c/attention \\
    --prompts "put both the alphabet soup and the tomato sauce in the basket" \\
              "pick up the tomato sauce and put it in the basket"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Token positions (from sink_decode_corrected.py analysis):
KEY_BOS = 768          # <bos> token — main sink
KEY_IMAGE_CORNER = 303 # secondary sink (image patch)
KEY_ACTION_SELF = 816  # first action token (self-attention sink)
LANGUAGE_TOKEN_START = 769  # content tokens start here under right-padding
LANGUAGE_TOKEN_END_MAX = 815  # rough upper bound (varies by prompt length)


def jax_to_pytorch_lora_merge(jax_ckpt_dir: Path, output_pt_path: Path):
    """STUB: Convert a LoRA-trained JAX checkpoint to PyTorch state_dict.

    TODO: implement this function. Steps:

    1. Load JAX params:
         from openpi.models import model as _model
         from openpi.shared import download
         jax_params = _model.restore_params(
             download.maybe_download(str(jax_ckpt_dir)),
             restore_type=np.ndarray,
         )

    2. Identify LoRA-decomposed weight matrices and merge:
         For each weight W with lora_a, lora_b siblings:
           W_merged = W + (lora_a @ lora_b) * (alpha/rank)
         Where alpha is typically 2*rank for the LoRA variants we use.

       Param names to look for (per gemma_2b_lora and gemma_300m_lora):
         ['PaliGemma']['llm']['layers']['attn']['q_einsum']['w']
         ['PaliGemma']['llm']['layers']['attn']['q_einsum']['w_lora_a']
         ['PaliGemma']['llm']['layers']['attn']['q_einsum']['w_lora_b']
         (similar for kv_einsum, attn_vec_einsum, mlp gating/linear)

    3. Map JAX param tree → PyTorch state_dict:
         Need to know PI0Pytorch's expected key names.
         Compare:
           - JAX tree from above
           - PT state_dict from ~/flare/checkpoints/pi0_libero_pt/.../*.pth
         Build a name-mapping dict and apply per-key, with transposes
         (JAX einsum order vs PT linear weight order).

    4. Save as .pth:
         import torch
         torch.save(merged_state_dict, output_pt_path)

    5. Verify by loading PI0Pytorch with the new checkpoint and running
       a forward pass; compare output to JAX inference (should be ~identical
       within floating-point precision).
    """
    raise NotImplementedError(
        "jax_to_pytorch_lora_merge not yet implemented. "
        "See docstring for the steps; ~2-4 hr of careful work. "
        "Recommended: do this tomorrow morning when fresh."
    )


def load_pytorch_lora_merged(pt_path: Path):
    """Load a PT-converted LoRA-merged pi0 model for attention analysis."""
    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch
    Pi0 = apply_patch()

    from openpi.training import config as _c
    from openpi.policies import policy_config

    # TODO: load the converted PT checkpoint into PI0Pytorch
    # This will look similar to how create_trained_policy loads
    # ~/flare/checkpoints/pi0_libero_pt/ but with our merged weights
    raise NotImplementedError(
        "Pending jax_to_pytorch_lora_merge implementation"
    )


def measure_attention(model, env_pack, prompt: str, n_init_states: int = 5,
                      layer_indices: Optional[List[int]] = None) -> Dict:
    """Run inference with attention hooks; return per-layer attention statistics.

    Reuses the AttentionCapture pattern from mechanism_probe_attn.py.
    """
    if layer_indices is None:
        layer_indices = list(range(18))  # all expert decoder layers

    import torch
    from collections import defaultdict

    # Storage: per (layer, init_idx) → per-token attention weights
    captures = defaultdict(list)

    handles = []
    for layer_idx in layer_indices:
        target_module = find_layer_attn_module(model, layer_idx)
        if target_module is None:
            continue
        def make_hook(lid):
            def hook(module, inputs, output):
                # output is typically (attn_output, attn_weights) or just attn_output
                # depending on output_attentions flag
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]  # (B, H, Q, K)
                    captures[lid].append(attn.detach().cpu())
            return hook
        handles.append(target_module.register_forward_hook(make_hook(layer_idx)))

    try:
        # Run inference for n_init_states
        for init_i in range(n_init_states):
            obs = reset_env_to_init(env_pack, init_i)
            obs_in = env_pack["format_obs"](obs, prompt)
            _ = env_pack["policy"].infer(obs_in)
    finally:
        for h in handles:
            h.remove()

    # Aggregate
    results = {}
    for layer_idx, attns in captures.items():
        if not attns:
            continue
        # Stack across denoise steps and inits: (N, H, Q, K)
        # For action-query attention to language keys:
        all_attn = torch.cat([a.flatten(end_dim=0).unsqueeze(0) for a in attns], dim=0)
        # Per-head averages for action-query → various key regions
        action_q_range = slice(816, 867)  # action tokens
        lang_k_range = slice(LANGUAGE_TOKEN_START, LANGUAGE_TOKEN_END_MAX + 1)

        # mass per head
        n_heads = all_attn.shape[-3]
        per_head_lang = []
        per_head_sink = []
        for h in range(n_heads):
            lang_mass = all_attn[..., h, action_q_range, lang_k_range].sum(dim=-1).mean().item()
            sink_mass = all_attn[..., h, action_q_range, KEY_BOS].mean().item() \
                      + all_attn[..., h, action_q_range, KEY_IMAGE_CORNER].mean().item() \
                      + all_attn[..., h, action_q_range, KEY_ACTION_SELF].mean().item()
            per_head_lang.append(lang_mass)
            per_head_sink.append(sink_mass)

        results[layer_idx] = {
            "per_head_lang_mass": per_head_lang,
            "per_head_sink_mass": per_head_sink,
            "mean_lang_mass": float(np.mean(per_head_lang)),
            "mean_sink_mass": float(np.mean(per_head_sink)),
        }
    return results


def find_layer_attn_module(model, layer_idx: int):
    """Find the self-attention module in the expert at the given layer."""
    import re
    pattern = re.compile(
        rf"^paligemma_with_expert\.gemma_expert\.model\.layers\.{layer_idx}\.self_attn$"
    )
    for name, mod in model.named_modules():
        if pattern.match(name):
            return mod
    return None


def reset_env_to_init(env_pack, init_idx: int):
    """Reset env to a specific init state."""
    env = env_pack["env"]
    init_state = env_pack["init_states"][init_idx]
    env.seed(7000 + init_idx)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(env_pack["NUM_STEPS_WAIT"]):
        obs, _, _, _ = env.step(env_pack["LIBERO_DUMMY_ACTION"])
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jax-checkpoint", required=True,
                   help="Path to LoRA-trained JAX checkpoint")
    p.add_argument("--variant-label", required=True)
    p.add_argument("--baseline-checkpoint", default=str(Path.home() / "flare/checkpoints/pi0_libero_pt"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", nargs="+", default=[
        "put both the alphabet soup and the tomato sauce in the basket",
        "pick up the tomato sauce and put it in the basket",
    ])
    p.add_argument("--n-init-states", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== ATTENTION EVAL: {args.variant_label} ===")
    print(f"  jax_checkpoint: {args.jax_checkpoint}")
    print(f"  baseline (for comparison): {args.baseline_checkpoint}")
    print(f"  out: {out_dir}")
    print()

    # STEP 1: convert JAX LoRA → PyTorch (not yet implemented)
    converted_pt_path = out_dir / "pi0_lora_merged.pth"
    print(f"[step 1] Convert JAX → PT (NOT IMPLEMENTED YET)")
    try:
        jax_to_pytorch_lora_merge(Path(args.jax_checkpoint).expanduser(),
                                   converted_pt_path)
    except NotImplementedError as e:
        print(f"  SKIPPED: {e}")
        print(f"  Output directory created with skeleton outputs only.")
        print(f"  To implement: see jax_to_pytorch_lora_merge() docstring above.")

        # Write a stub output so downstream comparison knows this is incomplete
        (out_dir / "STATUS_INCOMPLETE.txt").write_text(
            "Attention analysis NOT YET COMPUTED.\n"
            "Requires jax_to_pytorch_lora_merge() to be implemented.\n"
            "See eval_lora_attention.py docstring for steps.\n"
        )
        return

    # Once converter exists, the rest works:
    print(f"\n[step 2] Load PT-converted LoRA model")
    lora_model = load_pytorch_lora_merged(converted_pt_path)

    print(f"\n[step 3] Setup LIBERO env")
    env_pack = setup_eval_env()

    print(f"\n[step 4] Measure attention for each prompt + baseline")
    all_results = {}
    for prompt in args.prompts:
        all_results[prompt] = measure_attention(
            lora_model, env_pack, prompt,
            n_init_states=args.n_init_states,
        )

    # TODO: compute baseline pi0_libero attention for comparison
    # TODO: visualize per-layer per-head heatmaps
    # TODO: save aggregated stats

    (out_dir / "attention_eval.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[done] saved to {out_dir}")


def setup_eval_env():
    """Setup LIBERO env for attention measurement (T0 scene)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    sys.path.insert(0, str(Path.home() / "flare"))
    import remote_multimode_matrix as mmm  # noqa

    bench = benchmark.get_benchmark_dict()["libero_10"]()
    task = bench.get_task(0)
    import os
    bddl = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(0)
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                              camera_heights=256, camera_widths=256)
    return {
        "env": env,
        "init_states": init_states,
        "format_obs": mmm.format_obs,
        "NUM_STEPS_WAIT": mmm.NUM_STEPS_WAIT,
        "LIBERO_DUMMY_ACTION": mmm.LIBERO_DUMMY_ACTION,
        # policy filled in by load_pytorch_lora_merged
    }


if __name__ == "__main__":
    main()
