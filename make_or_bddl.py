"""make_or_bddl.py — produce a multimode-variant BDDL from a stock LIBERO BDDL.

Replaces:
  - the `(:language ...)` line with a new prompt
  - the `(:goal ...)` block with a new predicate (typically wrapping conjuncts
    in `(Or ...)` to create any-of-N success semantics)

The init regions, objects, and fixtures are preserved — only the language and
goal predicate change. So scene topology + the script's init_states from the
original task still apply.

Example for KITCHEN_SCENE8 (2 moka pots → "any pot on the stove" succeeds):

    python make_or_bddl.py \\
        --input  /home/williamchung/flare/openpi/third_party/libero/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl \\
        --output /home/williamchung/flare/custom_bddls/KITCHEN_SCENE8_put_a_moka_pot_OR.bddl \\
        --new-language "put a moka pot on the stove" \\
        --new-goal "(And (Or (On moka_pot_1 flat_stove_1_cook_region) (On moka_pot_2 flat_stove_1_cook_region)) (Turnon flat_stove_1))"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_balanced(text: str, start_idx: int) -> int:
    """Given the index of '(', return the index just past its matching ')'."""
    if text[start_idx] != "(":
        raise ValueError(f"Expected '(' at index {start_idx}, got {text[start_idx]!r}")
    depth = 0
    for i in range(start_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("Unbalanced parens — couldn't find matching ')'")


def replace_section(text: str, section: str, new_inner: str) -> str:
    """Replace a top-level `(:<section> ...)` block with new contents.

    The section is matched anywhere in the BDDL; the matching outer `)` is
    found by paren-balancing, so multiline blocks (like (:goal (And ...)))
    are handled correctly.
    """
    pattern = rf"\(:\s*{re.escape(section)}\b"
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Section {section!r} not found in BDDL")
    start = m.start()
    end = find_balanced(text, start)
    replacement = f"(:{section} {new_inner})"
    return text[:start] + replacement + text[end:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source BDDL path")
    p.add_argument("--output", required=True, help="Output BDDL path")
    p.add_argument("--new-language", required=False, default=None,
                   help="New language string (inserted literally — no quotes "
                        "added because LIBERO's BDDL writes language as bare tokens). "
                        "If omitted, original language is preserved.")
    p.add_argument("--new-goal", required=False, default=None,
                   help="New goal predicate as a fully-parenthesized PDDL expr, "
                        "e.g.  '(And (Or (On obj_1 region) (On obj_2 region)) (Turnon stove))'. "
                        "If omitted, original goal is preserved.")
    args = p.parse_args()

    if args.new_language is None and args.new_goal is None:
        p.error("at least one of --new-language or --new-goal must be provided")

    text = Path(args.input).read_text()
    if args.new_language is not None:
        text = replace_section(text, "language", args.new_language)
    if args.new_goal is not None:
        text = replace_section(text, "goal", args.new_goal)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    print(f"Saved: {out}")
    print()
    print("--- Final goal block ---")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("(:goal") or s.startswith("(:language"):
            print(f"  {line.rstrip()}")
    print("--- End ---")


if __name__ == "__main__":
    main()
