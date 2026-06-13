"""audit_counterfactual_data.py — verify whether existing LIBERO demos form a
valid counterfactual dataset for CF-LoRA training.

The CF-LoRA hypothesis: if we train pi0 on demos where the SAME scene template
+ DIFFERENT prompts → DIFFERENT actions, the model is forced to use prompts
causally to predict actions.

BUT this only works if the model can't cheat via non-language cues like:
  - object presence/absence (some tasks have only the target object visible)
  - systematic pose differences across tasks
  - different init-state distributions
  - subtle visual scene differences

This script audits all of these.

================================================================================
AUDIT QUESTIONS
================================================================================

Q1. OBJECT PRESENCE: Does each LIVING_ROOM_SCENE2 task have all 5 candidate
    objects in its scene, or just the target object? If only the target is
    present, the model can solve via visual saliency without using language.

Q2. POSE OVERLAP: For objects that ARE in multiple tasks, do their initial
    positions overlap across tasks? If each task places the target at a
    unique position, the model can cheat by spatial prior.

Q3. FIRST-MOVED OBJECT: For each demo, which object does the gripper actually
    approach and pick up first? This is the ground-truth label for training
    and eval.

Q4. BASELINE PI0 PROMPT-SWAP TEST: (already answered by EXP 3 / EXP 4)
    Does baseline pi0 follow prompt swaps on single-object tasks? Answer: NO.
    Single-object scenes also have bypass.

================================================================================
USAGE
================================================================================

python3 audit_counterfactual_data.py \\
    --datasets-root ~/flare/openpi/third_party/libero/libero/datasets \\
    --scene LIVING_ROOM_SCENE2 \\
    --out-dir ~/flare/results/cf_audit_scene2

This will:
  1. Find all HDF5s in libero_10 and libero_90 matching --scene
  2. Compute object-presence summary
  3. Compute per-object pose distributions across tasks
  4. Identify the first-moved object per demo
  5. Write a JSON report + a verdict on counterfactual validity
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


# LIBERO objects we care about for LIVING_ROOM_SCENE2
# These are the unique nouns extractable from libero_10/90 task prompts
LIVING_ROOM_OBJECTS = [
    "alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "milk",
    "orange_juice", "ketchup", "basket", "salad_dressing", "chocolate_pudding",
]


def discover_scene_tasks(datasets_root: Path, scene: str) -> List[Tuple[str, Path, str]]:
    """Find all LIBERO HDF5 demos matching the given scene name.

    Returns: list of (suite, path, inferred_prompt) tuples.
    """
    tasks = []
    for suite in ["libero_10", "libero_90", "libero_object"]:
        suite_dir = datasets_root / suite
        if not suite_dir.exists():
            continue
        for hdf5_path in sorted(suite_dir.glob(f"*{scene}*demo.hdf5")):
            # Infer prompt from filename
            name = hdf5_path.stem.replace("_demo", "")
            # Strip scene prefix if present
            if scene in name:
                prompt = name.split(scene + "_", 1)[1] if scene + "_" in name else name
            else:
                prompt = name
            prompt = prompt.replace("_", " ")
            tasks.append((suite, hdf5_path, prompt))
        # For libero_object (no scene prefix in name), include all
        if suite == "libero_object":
            for hdf5_path in sorted(suite_dir.glob("*demo.hdf5")):
                if hdf5_path in [t[1] for t in tasks]:
                    continue  # avoid duplicates
                name = hdf5_path.stem.replace("_demo", "").replace("_", " ")
                # Skip — libero_object uses floor scene, not LIVING_ROOM
                # (Keeping the iteration in case we want it later)
    return tasks


def get_object_positions(obs_group: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
    """Extract object positions at frame_idx from the obs group.

    LIBERO obs typically has per-object keys like 'tomato_sauce_1_pos'.
    Returns a dict {object_name: np.array(xyz)}.
    """
    positions = {}
    for key in obs_group.keys():
        if key.endswith("_pos") and not key.startswith("ee"):
            # Strip suffix to get object name
            name = re.sub(r"_\d+_pos$", "", key)
            try:
                pos = obs_group[key][frame_idx]
                if hasattr(pos, "shape") and pos.shape[-1] >= 3:
                    positions[name] = pos[:3].astype(np.float32)
            except Exception:
                pass
    return positions


def identify_first_moved_object(demo: h5py.Group, threshold: float = 0.05) -> str:
    """Find the first object whose position changes by more than `threshold` meters
    from its initial position.

    Returns the object name, or 'none' if no object moves.
    """
    obs = demo["obs"]

    # Initial positions
    init_positions = get_object_positions(obs, 0)
    if not init_positions:
        return "<no object positions found>"

    # Trajectory length
    T = len(demo["actions"])
    # Check at multiple time points
    check_points = [int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]

    for t in check_points:
        current_positions = get_object_positions(obs, t)
        for name, init_pos in init_positions.items():
            if name not in current_positions:
                continue
            displacement = np.linalg.norm(current_positions[name] - init_pos)
            if displacement > threshold:
                return name
    return "none"


def audit_hdf5(path: Path, max_demos: int = 50) -> Dict:
    """Audit a single HDF5 file: object presence, init positions, first-moved labels."""
    result = {
        "path": str(path),
        "n_demos_checked": 0,
        "objects_in_scene": [],
        "first_moved_distribution": {},
        "object_init_positions": {},  # {obj_name: mean_xyz}
        "object_init_pos_stds": {},   # {obj_name: std_xyz}
    }
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        demo_keys = demo_keys[:max_demos]
        result["n_demos_checked"] = len(demo_keys)

        # Objects in scene: check demo_0's obs keys
        first_obs = f["data"][demo_keys[0]]["obs"]
        all_init_positions = get_object_positions(first_obs, 0)
        result["objects_in_scene"] = sorted(all_init_positions.keys())

        # First-moved distribution
        first_moved_counts = {}
        per_obj_inits = {name: [] for name in all_init_positions.keys()}

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            obs = demo["obs"]

            # First moved
            fm = identify_first_moved_object(demo)
            first_moved_counts[fm] = first_moved_counts.get(fm, 0) + 1

            # Init positions for each object
            inits = get_object_positions(obs, 0)
            for name, pos in inits.items():
                if name in per_obj_inits:
                    per_obj_inits[name].append(pos)

        result["first_moved_distribution"] = first_moved_counts

        # Compute per-object init position stats
        for name, positions in per_obj_inits.items():
            if positions:
                arr = np.stack(positions)
                result["object_init_positions"][name] = arr.mean(axis=0).tolist()
                result["object_init_pos_stds"][name] = arr.std(axis=0).tolist()
    return result


def compute_pose_overlap(per_task_audits: List[Dict]) -> Dict:
    """For each object that appears in multiple tasks, compute cross-task pose
    centroid distances and overlap ratio."""
    # Build {object: {task: mean_xyz}}
    obj_per_task = {}
    for task in per_task_audits:
        task_name = Path(task["path"]).stem.replace("_demo", "")
        for obj, mean_xyz in task["object_init_positions"].items():
            obj_per_task.setdefault(obj, {})[task_name] = (
                np.array(mean_xyz),
                np.array(task["object_init_pos_stds"][obj]),
            )

    overlaps = {}
    for obj, tasks_dict in obj_per_task.items():
        if len(tasks_dict) < 2:
            continue
        task_names = sorted(tasks_dict.keys())
        pairs = []
        for i in range(len(task_names)):
            for j in range(i + 1, len(task_names)):
                m_i, s_i = tasks_dict[task_names[i]]
                m_j, s_j = tasks_dict[task_names[j]]
                dist = float(np.linalg.norm(m_i - m_j))
                max_within = float(max(s_i.mean(), s_j.mean()))
                ratio = dist / max(max_within, 1e-9)
                pairs.append({
                    "task_a": task_names[i],
                    "task_b": task_names[j],
                    "centroid_dist": dist,
                    "max_within_std": max_within,
                    "ratio": ratio,
                })
        overlaps[obj] = {
            "n_tasks_with_object": len(tasks_dict),
            "pair_comparisons": pairs,
            "max_ratio": max(p["ratio"] for p in pairs),
            "mean_ratio": float(np.mean([p["ratio"] for p in pairs])),
        }
    return overlaps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-root", required=True,
                   help="Path to libero/libero/datasets/")
    p.add_argument("--scene", default="LIVING_ROOM_SCENE2",
                   help="Scene name to audit (e.g. LIVING_ROOM_SCENE2)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-demos", type=int, default=50)
    args = p.parse_args()

    datasets_root = Path(args.datasets_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 80)
    print(f"COUNTERFACTUAL DATASET AUDIT for scene={args.scene}")
    print(f"=" * 80)

    # 1. Discover tasks for this scene
    tasks = discover_scene_tasks(datasets_root, args.scene)
    print(f"\n[1/3] Found {len(tasks)} tasks matching scene '{args.scene}':")
    for suite, path, prompt in tasks:
        print(f"  [{suite}] {path.name}")
        print(f"           inferred prompt: {prompt!r}")

    if len(tasks) < 2:
        print(f"\n[WARN] Need at least 2 tasks for a counterfactual audit.")
        return

    # 2. Audit each task
    print(f"\n[2/3] Auditing each task (max_demos={args.max_demos}) ...")
    per_task_audits = []
    for suite, path, prompt in tasks:
        print(f"\n  {path.name}")
        try:
            audit = audit_hdf5(path, max_demos=args.max_demos)
            audit["suite"] = suite
            audit["inferred_prompt"] = prompt
            per_task_audits.append(audit)

            print(f"    objects_in_scene: {audit['objects_in_scene']}")
            print(f"    first-moved: {audit['first_moved_distribution']}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            continue

    # 3. Compute cross-task overlap
    print(f"\n[3/3] Computing cross-task pose overlap ...")
    overlap = compute_pose_overlap(per_task_audits)

    print(f"\n  Per-object cross-task pose overlap (lower ratio = more overlap):")
    print(f"  {'object':<25s} {'n_tasks':>7s} {'mean_ratio':>12s} {'max_ratio':>12s}  verdict")
    print(f"  {'-' * 25} {'-' * 7} {'-' * 12} {'-' * 12}  {'-' * 30}")
    for obj, stats in sorted(overlap.items(), key=lambda x: -x[1]["max_ratio"]):
        if stats["max_ratio"] < 3:
            verdict = "GOOD overlap (CF-viable)"
        elif stats["max_ratio"] < 10:
            verdict = "moderate (cheat risk)"
        else:
            verdict = "POOR overlap (likely cheat)"
        print(f"  {obj:<25s} {stats['n_tasks_with_object']:>7d} "
              f"{stats['mean_ratio']:>12.1f} {stats['max_ratio']:>12.1f}  {verdict}")

    # 4. Object presence summary
    print(f"\n  Object presence across tasks:")
    all_objects = set()
    for audit in per_task_audits:
        all_objects.update(audit["objects_in_scene"])
    presence_matrix = {}
    for obj in sorted(all_objects):
        present_in = []
        for audit in per_task_audits:
            if obj in audit["objects_in_scene"]:
                present_in.append(Path(audit["path"]).stem.replace("_demo", ""))
        presence_matrix[obj] = {
            "n_tasks_present": len(present_in),
            "total_tasks": len(per_task_audits),
            "tasks": present_in,
        }
        print(f"    {obj:<25s} in {len(present_in)}/{len(per_task_audits)} tasks")

    # 5. Overall verdict
    print(f"\n=== OVERALL VERDICT ===")
    n_with_good_overlap = sum(1 for o in overlap.values() if o["max_ratio"] < 3)
    n_with_moderate = sum(1 for o in overlap.values() if 3 <= o["max_ratio"] < 10)
    n_with_poor = sum(1 for o in overlap.values() if o["max_ratio"] >= 10)
    print(f"  Objects appearing in multiple tasks: {len(overlap)}")
    print(f"    GOOD overlap (ratio < 3):     {n_with_good_overlap}")
    print(f"    MODERATE (3 ≤ ratio < 10):    {n_with_moderate}")
    print(f"    POOR overlap (ratio ≥ 10):    {n_with_poor}")

    if n_with_poor > n_with_good_overlap + n_with_moderate:
        verdict = "AT RISK — most objects have systematic position differences across tasks. Model likely to cheat via spatial prior. CF-LoRA may need attention regularization or same-state augmentation."
    elif n_with_moderate > 0:
        verdict = "PARTIAL — some objects overlap well, others differ systematically. Natural CF-LoRA likely partial; attention regularization may be needed."
    else:
        verdict = "GOOD — object poses overlap well across tasks. Natural CF-LoRA has a real chance."
    print(f"\n  {verdict}")

    # Save full report
    report = {
        "scene": args.scene,
        "n_tasks": len(per_task_audits),
        "per_task_audits": per_task_audits,
        "cross_task_pose_overlap": overlap,
        "object_presence_matrix": presence_matrix,
        "verdict": verdict,
        "verdict_counts": {
            "good": n_with_good_overlap,
            "moderate": n_with_moderate,
            "poor": n_with_poor,
        },
    }
    out_path = out_dir / f"cf_audit_{args.scene}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Full report saved: {out_path}")


if __name__ == "__main__":
    main()
