"""enumerate_libero_prompts.py — list all LIBERO task prompts to verify which
prompts pi0_libero has actually seen during training.

Context: the v10 steering result uses prompt0="put both the alphabet soup and
the tomato sauce in the basket" (the T0 training prompt) vs prompt1="put the
tomato sauce in the basket" (probe contrast). If prompt1 has NEVER appeared in
the training distribution, the probe direction is just a linguistic axis (not
a behavioral one), which sharpens the steering interpretation: the 20% mode-B
reach is the model's motor primitives surfacing under perturbation, not
trained mode-B behavior being unlocked.

This script enumerates EVERY LIBERO suite's task prompts and flags any that
match patterns of interest. Run before drawing conclusions about steering.

Usage (on remote):
    TORCH_COMPILE_DISABLE=1 ~/flare/openpi/.venv/bin/python \\
        ~/flare/enumerate_libero_prompts.py \\
        --search "tomato" "alphabet" "put the" "in the basket" \\
        --out ~/flare/results/libero_prompts.json
"""
import argparse
import json
import re
import sys
from pathlib import Path


def get_task_prompt(task_obj):
    """Try multiple known fields for the language instruction."""
    for attr in ("language", "task_emb", "instruction", "description",
                  "task_name", "name"):
        val = getattr(task_obj, attr, None)
        if val is not None and not callable(val):
            return str(val)
    return "(no language attribute found)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+",
                   default=["libero_spatial", "libero_object",
                            "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--search", nargs="+",
                   default=["tomato", "alphabet", "put the"])
    p.add_argument("--out", default="~/flare/results/libero_prompts.json")
    p.add_argument("--print-all", action="store_true",
                   help="Print every task prompt (default: only matches)")
    args = p.parse_args()

    from libero.libero import benchmark
    bench_dict = benchmark.get_benchmark_dict()
    print(f"[info] available LIBERO benchmark suites: {list(bench_dict.keys())}\n")

    all_prompts = {}
    for suite in args.suites:
        if suite not in bench_dict:
            print(f"  [skip] {suite!r} not in benchmark dict")
            continue
        try:
            b = bench_dict[suite]()
        except Exception as e:
            print(f"  [skip] {suite!r} init failed: {type(e).__name__}: {e}")
            continue
        try:
            n_tasks = b.n_tasks
        except Exception:
            n_tasks = len(b.tasks) if hasattr(b, "tasks") else 0

        suite_prompts = []
        for i in range(n_tasks):
            try:
                t = b.get_task(i)
            except Exception as e:
                print(f"  [{suite} {i}] get_task failed: {e}")
                continue
            prompt = get_task_prompt(t)
            name = getattr(t, "name", f"task_{i}")
            suite_prompts.append({
                "task_id": i, "name": name, "language": prompt,
            })
        all_prompts[suite] = suite_prompts

        print(f"=== {suite}: {n_tasks} tasks ===")
        for ent in suite_prompts:
            highlights = [s for s in args.search
                           if s.lower() in ent["language"].lower()]
            if highlights or args.print_all:
                marker = f" *** matches: {highlights} ***" if highlights else ""
                print(f"  [{ent['task_id']:>3d}] {ent['name']:<50s}  "
                      f"{ent['language']!r}{marker}")
        print()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_prompts, indent=2))
    print(f"[done] saved {out}\n")

    # Per-search-term summary
    print(f"=== SEARCH SUMMARY ===")
    for s in args.search:
        sl = s.lower()
        print(f"\n  Pattern: {s!r}")
        total = 0
        for suite, prompts in all_prompts.items():
            matches = [p for p in prompts if sl in p["language"].lower()]
            if matches:
                print(f"    {suite}: {len(matches)} match(es)")
                for m in matches:
                    print(f"      [{m['task_id']:>3d}] {m['language']!r}")
                total += len(matches)
        print(f"    -- total across suites: {total}")

    # Specific verification
    print(f"\n=== KEY QUESTION: does 'put the tomato sauce in the basket' "
          f"appear anywhere? ===")
    target = "put the tomato sauce in the basket"
    found = False
    for suite, prompts in all_prompts.items():
        for p in prompts:
            if target.lower() in p["language"].lower():
                print(f"  YES: {suite} [{p['task_id']}] {p['language']!r}")
                found = True
    if not found:
        print(f"  NO: 'put the tomato sauce in the basket' is NOT in any "
              f"LIBERO suite enumerated.")
        print(f"  → confirms prompt1 is OOD; steering interpretation = "
              f"linguistic-axis perturbation, not learned-mode access.")


if __name__ == "__main__":
    main()
