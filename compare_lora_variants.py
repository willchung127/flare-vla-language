"""compare_lora_variants.py — aggregate headline metrics across all trained
LoRA variants and produce comparison report + plots.

Reads each variant's behavioral_eval.json + velocity_eval.json from
~/flare/results/lora_ablation/{variant}/ and produces:
  - headline_table.md           (markdown table for paper)
  - first_moved_compare.png     (per-prompt first-moved across variants)
  - velocity_cos_compare.png    (LoRA cos vs baseline across variants)
  - success_rate_compare.png    (per-prompt success across variants)
  - comparison_summary.json     (full data)
  - comparison_report.txt       (paper-ready text)

================================================================================
USAGE
================================================================================

# Compare all variants in default directory
python3 compare_lora_variants.py \\
    --ablation-dir ~/flare/results/lora_ablation \\
    --variants variant_a variant_a_prime variant_c \\
    --out-dir ~/flare/results/lora_ablation/comparison

# Or auto-discover all variant_* dirs
python3 compare_lora_variants.py \\
    --ablation-dir ~/flare/results/lora_ablation \\
    --auto-discover \\
    --out-dir ~/flare/results/lora_ablation/comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


HEADLINE_METRICS = [
    ("tomato_rate_on_tomato_prompt",
     "Tomato-first | tomato_only prompt",
     "Baseline = 0%. Higher = better."),
    ("alphabet_rate_on_canonical",
     "Alphabet-first | T0 canonical prompt",
     "Baseline = 87%. Should stay high (preservation)."),
    ("tomato_rate_on_reordered",
     "Tomato-first | reordered prompt",
     "If model parses word order, this rises."),
    ("alphabet_rate_on_reordered",
     "Alphabet-first | reordered prompt",
     "If model rote-matches, this stays high."),
    ("alphabet_rate_on_alphabet_only",
     "Alphabet-first | alphabet_only prompt",
     "Sanity: trained single-target → matches."),
]


def load_variant_data(variant_dir: Path) -> Dict:
    """Load all eval data for a variant."""
    data = {"variant_dir": str(variant_dir)}

    # Behavioral
    behav_path = variant_dir / "behavioral_eval.json"
    if behav_path.exists():
        with behav_path.open() as f:
            data["behavioral"] = json.load(f)
    elif (variant_dir / "headline_metrics.json").exists():
        with (variant_dir / "headline_metrics.json").open() as f:
            data["headline_only"] = json.load(f)

    # Velocity
    vel_paths = [
        variant_dir / "velocity_probes" / "velocity_eval.json",
        variant_dir / "velocity_eval.json",
    ]
    for vp in vel_paths:
        if vp.exists():
            with vp.open() as f:
                data["velocity"] = json.load(f)
            break

    # Attention (may be missing if not yet implemented)
    attn_path = variant_dir / "attention" / "attention_eval.json"
    if attn_path.exists():
        with attn_path.open() as f:
            data["attention"] = json.load(f)

    return data


def collect_headline(variant_data: Dict[str, Dict]) -> Dict:
    """Pull headline metrics from each variant's behavioral data."""
    headlines = {}
    for variant, data in variant_data.items():
        h = None
        if "behavioral" in data:
            h = data["behavioral"].get("headline")
        elif "headline_only" in data:
            h = data["headline_only"]
        if h:
            headlines[variant] = h
    return headlines


def render_markdown_table(headlines: Dict[str, Dict]) -> str:
    """Build a markdown table comparing variants."""
    variants = list(headlines.keys())
    if not variants:
        return "No variants with headline metrics found.\n"

    lines = ["# LoRA Ablation: Headline Metrics\n"]
    lines.append(f"Comparing {len(variants)} variant(s):\n")
    for v in variants:
        lines.append(f"  - **{v}**")
    lines.append("")

    # Build table
    header = "| Metric | " + " | ".join(variants) + " |"
    sep = "|" + "---|" * (len(variants) + 1)
    lines.append(header)
    lines.append(sep)
    for key, name, _ in HEADLINE_METRICS:
        row = f"| {name} |"
        for v in variants:
            val = headlines[v].get(key)
            if val is None:
                row += " — |"
            else:
                row += f" {val*100:.0f}% |"
        lines.append(row)

    # Notes
    lines.append("\n## Notes\n")
    for key, name, note in HEADLINE_METRICS:
        lines.append(f"- **{name}**: {note}")
    return "\n".join(lines) + "\n"


def make_first_moved_plot(headlines: Dict[str, Dict], variant_data: Dict[str, Dict],
                           out_path: Path):
    """Per-prompt first-moved distribution, with one cluster of bars per prompt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Need behavioral data to get distributions per prompt
    valid_variants = [v for v in headlines.keys() if "behavioral" in variant_data[v]]
    if not valid_variants:
        print(f"[warn] no variants with full behavioral data for first-moved plot")
        return

    # Get all prompts (assume all variants share the same prompt set)
    first_variant_data = variant_data[valid_variants[0]]["behavioral"]
    prompts = list(first_variant_data["summary"].keys())

    # For each prompt: stacked bar showing first-moved distribution across variants
    n_prompts = len(prompts)
    n_variants = len(valid_variants)

    # Collect all unique objects across all variants
    all_objs = set()
    for v in valid_variants:
        for prompt in prompts:
            dist = variant_data[v]["behavioral"]["summary"][prompt]["first_moved_distribution"]
            all_objs.update(dist.keys())
    all_objs = sorted(all_objs)

    fig, axes = plt.subplots(n_prompts, 1, figsize=(max(12, n_variants * 2), 3 * n_prompts))
    if n_prompts == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_objs)))
    for ax, prompt in zip(axes, prompts):
        counts = np.zeros((n_variants, len(all_objs)))
        for i, v in enumerate(valid_variants):
            dist = variant_data[v]["behavioral"]["summary"][prompt]["first_moved_distribution"]
            for j, obj in enumerate(all_objs):
                counts[i, j] = dist.get(obj, 0)

        bottom = np.zeros(n_variants)
        for j, obj in enumerate(all_objs):
            if counts[:, j].sum() == 0:
                continue
            ax.bar(valid_variants, counts[:, j], bottom=bottom,
                   label=obj, color=colors[j])
            bottom += counts[:, j]
        ax.set_title(f"prompt: {prompt}")
        ax.set_ylabel("# trials")
        if ax == axes[0]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    fig.suptitle("First-moved distribution across LoRA variants", y=1.001)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def make_velocity_plot(variant_data: Dict[str, Dict], out_path: Path):
    """Compare velocity cos (LoRA vs baseline) across variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find variants with velocity data
    valid = []
    for v, data in variant_data.items():
        if "velocity" in data and "comparison" in data["velocity"]:
            valid.append(v)
    if not valid:
        print(f"[warn] no variants with velocity comparison data")
        return

    # Extract pairs across variants
    all_pairs = set()
    for v in valid:
        all_pairs.update(variant_data[v]["velocity"]["comparison"].keys())
    pair_labels = sorted(all_pairs)

    n_pairs = len(pair_labels)
    n_variants = len(valid)
    width = 0.8 / (n_variants + 1)  # +1 for baseline

    fig, ax = plt.subplots(figsize=(max(10, n_pairs * 3), 5))
    x = np.arange(n_pairs)

    # Baseline bars
    baseline_vals = []
    for label in pair_labels:
        bl = None
        for v in valid:
            comp = variant_data[v]["velocity"]["comparison"].get(label, {})
            if comp.get("baseline_cos") is not None:
                bl = comp["baseline_cos"]
                break
        baseline_vals.append(bl if bl is not None else 0)
    ax.bar(x - 0.4 + width/2, baseline_vals, width,
           label="Baseline pi0_libero", color="gray")

    # Variant bars
    for i, v in enumerate(valid):
        vals = []
        for label in pair_labels:
            comp = variant_data[v]["velocity"]["comparison"].get(label, {})
            vals.append(comp.get("lora_cos", 0))
        offset = -0.4 + width/2 + width * (i + 1)
        ax.bar(x + offset, vals, width, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=15, ha="right")
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Velocity cos (real 7 dims)")
    ax.set_title("Velocity cos comparison: baseline vs LoRA variants")
    ax.axhline(1.0, ls="--", color="lightgray", lw=0.8)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def write_report(headlines: Dict[str, Dict], variant_data: Dict[str, Dict],
                  out_path: Path):
    """Paper-ready text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LoRA Ablation Comparison Report")
    lines.append("=" * 80)
    lines.append("")

    if not headlines:
        lines.append("No variants with headline metrics found.")
        out_path.write_text("\n".join(lines))
        return

    # Headline table
    lines.append("HEADLINE: Did LoRA fix the language bypass?\n")
    variants = list(headlines.keys())
    col_w = max(20, max(len(v) for v in variants) + 2)

    header = "  " + "Metric".ljust(50) + "  " + "  ".join(v.ljust(col_w) for v in variants)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key, name, note in HEADLINE_METRICS:
        row = "  " + name.ljust(50) + "  "
        for v in variants:
            val = headlines[v].get(key)
            row += (f"{val*100:.0f}%".ljust(col_w) if val is not None else "—".ljust(col_w)) + "  "
        lines.append(row)
    lines.append("")

    # Decision tree per variant
    lines.append("VERDICT PER VARIANT")
    lines.append("-" * 80)
    for v in variants:
        h = headlines[v]
        tomato = h.get("tomato_rate_on_tomato_prompt", 0)
        alpha_pres = h.get("alphabet_rate_on_canonical", 0)
        lines.append(f"\n  {v}:")
        lines.append(f"    Tomato-first on tomato prompt: {tomato*100:.0f}%")
        lines.append(f"    Alphabet-preservation: {alpha_pres*100:.0f}%")
        if tomato >= 0.5 and alpha_pres >= 0.7:
            lines.append(f"    → STRONG SUCCESS (>50% prompt control + preservation)")
        elif tomato >= 0.3:
            lines.append(f"    → PARTIAL SUCCESS (some prompt sensitivity)")
        elif tomato >= 0.1:
            lines.append(f"    → MINOR SHIFT (mostly bypass remains)")
        else:
            lines.append(f"    → NO IMPROVEMENT (bypass not addressed)")

    # Velocity comparison summary
    lines.append("\n\nVELOCITY DIVERGENCE")
    lines.append("-" * 80)
    for v in variants:
        if "velocity" not in variant_data[v]:
            continue
        lines.append(f"\n  {v}:")
        comp = variant_data[v]["velocity"].get("comparison", {})
        for pair_label, pair_data in comp.items():
            bl = pair_data.get("baseline_cos")
            lora = pair_data.get("lora_cos")
            if bl is not None and lora is not None:
                delta = lora - bl
                arrow = "↓ more diverged" if delta < -0.005 else "↑ less diverged" if delta > 0.005 else "≈ unchanged"
                lines.append(f"    {pair_label}: baseline {bl:.4f} → LoRA {lora:.4f}  ({arrow})")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-dir", default="~/flare/results/lora_ablation",
                   help="Directory containing variant subdirs")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Explicit variant subdir names")
    p.add_argument("--auto-discover", action="store_true",
                   help="Auto-find all variant_* subdirs in ablation-dir")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    ablation_dir = Path(args.ablation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir or ablation_dir / "comparison").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover variants
    if args.auto_discover or args.variants is None:
        variants = sorted(d.name for d in ablation_dir.iterdir()
                          if d.is_dir() and d.name.startswith("variant"))
    else:
        variants = args.variants

    if not variants:
        print(f"[error] no variants found in {ablation_dir}")
        return

    print(f"\n=== COMPARING {len(variants)} VARIANT(S) ===")
    for v in variants:
        print(f"  - {v}")

    # Load each
    variant_data = {}
    for v in variants:
        variant_data[v] = load_variant_data(ablation_dir / v)

    # Collect headlines
    headlines = collect_headline(variant_data)
    print(f"\n[load] {len(headlines)}/{len(variants)} variants have headline metrics")

    # Markdown table
    md = render_markdown_table(headlines)
    (out_dir / "headline_table.md").write_text(md)
    print(f"\n[wrote] headline_table.md")
    print(md)

    # Save full comparison JSON
    full = {
        "variants": variants,
        "ablation_dir": str(ablation_dir),
        "headlines": headlines,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(full, indent=2, default=str))

    # Plots
    print(f"\n[plots] generating...")
    try:
        make_first_moved_plot(headlines, variant_data,
                               out_dir / "first_moved_compare.png")
    except Exception as e:
        print(f"  first_moved plot failed: {e}")
    try:
        make_velocity_plot(variant_data, out_dir / "velocity_cos_compare.png")
    except Exception as e:
        print(f"  velocity plot failed: {e}")

    # Paper-ready report
    write_report(headlines, variant_data, out_dir / "comparison_report.txt")
    print(f"\n[wrote] comparison_report.txt")

    print(f"\n[done] all outputs in {out_dir}")


if __name__ == "__main__":
    main()
