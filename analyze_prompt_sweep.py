"""analyze_prompt_sweep.py — aggregate the 20-prompt sweep into one table.

Reads each ~/flare/results/multimode_matrix_<sweep_suffix>_<variant_name>/ODE_single.json
and computes first-moved-object distribution + success rate per variant.

Prints a single table grouped by category (A/B/C/D/E) so the structure is visible.

Usage:
    python3 analyze_prompt_sweep.py [--results-dir ~/flare/results] [--out-suffix promptsweep]
"""
import argparse
import json
from collections import Counter
from pathlib import Path


PROMPT_CATEGORIES = {
    # variant_name → category letter
    # A — reordering
    "T0_A1_reorder_both": "A",
    "T0_A2_reorder_nob": "A",
    "T0_A3_reorder_can": "A",
    "T0_A4_reorder_nothe": "A",
    "T0_A5_reorder_passive": "A",
    # B — single-target paraphrases
    "T0_B1_single_put": "B",
    "T0_B2_single_place": "B",
    "T0_B3_single_pickput": "B",
    "T0_B4_single_move": "B",
    "T0_B5_single_grab": "B",
    # C — explicit ordering
    "T0_C1_order_firstthen": "C",
    "T0_C2_order_startwith": "C",
    "T0_C3_order_goesfirst": "C",
    "T0_C4_order_before": "C",
    # D — negative phrasing
    "T0_D1_neg_not": "D",
    "T0_D2_neg_avoid": "D",
    "T0_D3_neg_dontmove": "D",
    # E — cross-task
    "T0_E1_cross_t1": "E",
    "T0_E2_cross_t1rev": "E",
    "T0_E3_cross_milk": "E",
}

CATEGORY_NAMES = {
    "A": "Reordering ('put both X and Y')",
    "B": "Single-target paraphrase",
    "C": "Explicit ordering ('first X then Y')",
    "D": "Negative phrasing ('not Y')",
    "E": "Cross-task template",
}

# Per-task default mode-A object (so we can compute "did pi0.5 flip from mode A?")
MODE_A_OBJECT = "alphabet_soup_1_pos"  # T0 default mode-A
MODE_B_OBJECT = "tomato_sauce_1_pos"  # T0 default mode-B target


def analyze_one(json_path: Path) -> dict:
    """Load one variant's JSON and compute summary metrics."""
    with open(json_path) as f:
        trials = json.load(f)
    if not trials:
        return None

    N = len(trials)
    n_succ = sum(int(t.get("success", False)) for t in trials)
    first_moved = []
    language = trials[0].get("language_used", "<unknown>")
    for t in trials:
        mo = t.get("moved_objects", [])
        if mo:
            first_moved.append(mo[0]["name"])
        else:
            first_moved.append("<none>")
    counter = Counter(first_moved)
    n_mode_a = counter.get(MODE_A_OBJECT, 0)
    n_mode_b = counter.get(MODE_B_OBJECT, 0)
    n_off_task = N - n_mode_a - n_mode_b - counter.get("<none>", 0)
    return {
        "N": N,
        "n_succ": n_succ,
        "succ_rate": n_succ / N if N else 0,
        "n_mode_a": n_mode_a,
        "n_mode_b": n_mode_b,
        "n_off_task": n_off_task,
        "mode_b_rate": n_mode_b / N if N else 0,
        "language": language,
        "first_moved_counter": dict(counter),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=str(Path.home() / "flare/results"))
    p.add_argument("--out-suffix", type=str, default="promptsweep",
                   help="The --out-suffix used at runtime; output dirs are "
                        "multimode_matrix_<out_suffix>_<variant_name>/")
    args = p.parse_args()

    results_dir = Path(args.results_dir).expanduser()

    print("=" * 96)
    print(f"  20-PROMPT SWEEP — Mode-B (tomato_sauce) first-moved rate on T0 (pi0.5)")
    print("=" * 96)
    print()
    fmt = "  {:<22s} | {:<48s} | {:>5s} | {:>5s} | {:>5s} | {:>5s} | {:>5s}"
    print(fmt.format("Variant", "Language", "N", "Succ%", "ModeA", "ModeB", "OffT"))
    print("  " + "─" * 94)

    per_category = {c: [] for c in CATEGORY_NAMES.keys()}

    for variant_stem, category in PROMPT_CATEGORIES.items():
        json_path = results_dir / f"multimode_matrix_{args.out_suffix}_{variant_stem}" / "ODE_single.json"
        if not json_path.exists():
            print(fmt.format(variant_stem, "(missing)", "-", "-", "-", "-", "-"))
            continue
        try:
            d = analyze_one(json_path)
        except Exception as e:
            print(fmt.format(variant_stem, f"(error: {e})", "-", "-", "-", "-", "-"))
            continue
        per_category[category].append((variant_stem, d))

    # Print grouped by category
    for cat, rows in per_category.items():
        print()
        print(f"  ── Category {cat}: {CATEGORY_NAMES[cat]} " + "─" * (40 - len(CATEGORY_NAMES[cat])))
        if not rows:
            print(f"    (no data for category {cat})")
            continue
        for variant_stem, d in rows:
            lang_short = d['language'][:48] if d['language'] else "<no lang>"
            print(fmt.format(
                variant_stem.replace("T0_", ""),
                lang_short,
                str(d['N']),
                f"{100*d['succ_rate']:.0f}%",
                f"{d['n_mode_a']}",
                f"{d['n_mode_b']}",
                f"{d['n_off_task']}",
            ))

    # Bottom-line summary
    print()
    print("=" * 96)
    total_b = sum(d['n_mode_b'] for rows in per_category.values() for _, d in rows)
    total_n = sum(d['N'] for rows in per_category.values() for _, d in rows)
    print(f"  Headline: {total_b}/{total_n} trials picked mode-B (tomato_sauce) first across all 20 prompts")
    # Per-category mode-B rates
    print()
    print("  Per-category mode-B rate:")
    for cat in sorted(per_category.keys()):
        rows = per_category[cat]
        if not rows:
            continue
        cat_b = sum(d['n_mode_b'] for _, d in rows)
        cat_n = sum(d['N'] for _, d in rows)
        rate = (100 * cat_b / cat_n) if cat_n else 0
        print(f"    {cat} ({CATEGORY_NAMES[cat]}): {cat_b}/{cat_n} = {rate:.1f}%")
    print("=" * 96)


if __name__ == "__main__":
    main()
