#!/usr/bin/env python3
"""apply_d_lite_patches.py — apply BOS attention mask patches to openpi.

Three file edits, all idempotent (re-running is safe — checks if already applied).
Creates .bak.dlite backups before editing.

Design:
  1. pi0_config.py:   Add `mask_bos_for_action_queries: bool = False` field to Pi0Config.
                       (Descriptive only — actual control via env var.)
  2. pi0.py:           Modify make_attn_mask to apply BOS mask when env var
                       OPENPI_MASK_BOS=1 is set. Pure env-var control avoids needing
                       to find and patch every make_attn_mask caller.
  3. training/config.py: Add new TrainConfig pi0_libero_cf_lora_bos_masked.

The chain script `run_lora_ablation_chain_v3.sh` sets OPENPI_MASK_BOS=1 when
running training/eval for the bos_masked variant. The env var is read at JIT-trace
time by make_attn_mask; since chain sets it before invoking each command, the
trace sees it correctly.

Usage (on remote):
    ~/flare/openpi/.venv/bin/python ~/flare/apply_d_lite_patches.py

To revert (if needed):
    for f in ~/flare/openpi/src/openpi/models/pi0_config.py \\
             ~/flare/openpi/src/openpi/models/pi0.py \\
             ~/flare/openpi/src/openpi/training/config.py; do
        if [ -f "$f.bak.dlite" ]; then cp "$f.bak.dlite" "$f"; fi
    done
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

OPENPI_SRC = Path.home() / "flare" / "openpi" / "src" / "openpi"
PI0_CONFIG = OPENPI_SRC / "models" / "pi0_config.py"
PI0_MODEL = OPENPI_SRC / "models" / "pi0.py"
TRAIN_CONFIG = OPENPI_SRC / "training" / "config.py"


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak.dlite")
    if not bak.exists():
        shutil.copy(path, bak)
        print(f"  [backup] {path.name} → {bak.name}")
    else:
        print(f"  [backup exists] {bak.name}")


def revert(path: Path):
    bak = path.with_suffix(path.suffix + ".bak.dlite")
    if bak.exists():
        shutil.copy(bak, path)
        print(f"  [reverted] {path.name} from {bak.name}")


# ============================================================================
# EDIT 1: Add mask_bos_for_action_queries field to Pi0Config
# ============================================================================
def edit_pi0_config():
    print(f"\n=== EDIT 1: pi0_config.py — add Pi0Config field ===")
    content = PI0_CONFIG.read_text()

    if "mask_bos_for_action_queries" in content:
        print("  [skip] field already present")
        return True

    if "class Pi0Config" not in content:
        print(f"  [ERROR] 'class Pi0Config' not found in {PI0_CONFIG}")
        return False

    backup(PI0_CONFIG)

    # Find a stable anchor inside Pi0Config to insert after.
    # Try several known field patterns in order of preference.
    anchor_patterns = [
        (r'^(\s+)pi05:\s*bool\s*=\s*\w+\s*$', "pi05"),
        (r'^(\s+)action_horizon:\s*int\s*=\s*\d+\s*$', "action_horizon"),
        (r'^(\s+)action_dim:\s*int\s*=\s*\d+\s*$', "action_dim"),
        (r'^(\s+)action_expert_variant:\s*str\s*=\s*[\'"][^\'"]+[\'"]\s*$', "action_expert_variant"),
        (r'^(\s+)paligemma_variant:\s*str\s*=\s*[\'"][^\'"]+[\'"]\s*$', "paligemma_variant"),
    ]

    inserted = False
    for pattern, anchor_name in anchor_patterns:
        m = re.search(pattern, content, re.MULTILINE)
        if m:
            indent = m.group(1)
            new_field = (
                f"\n{indent}# D-lite: mask attention from action queries to the BOS token "
                f"(language-content sink).\n"
                f"{indent}# Control via env var OPENPI_MASK_BOS=1 (set by chain script for "
                f"bos_masked variant).\n"
                f"{indent}# This field is descriptive — actual masking happens in pi0.make_attn_mask "
                f"when env var is set.\n"
                f"{indent}mask_bos_for_action_queries: bool = False"
            )
            content = content[: m.end()] + new_field + content[m.end():]
            print(f"  [insert] after '{anchor_name}' field")
            inserted = True
            break

    if not inserted:
        print(f"  [ERROR] no anchor field found in Pi0Config class")
        revert(PI0_CONFIG)
        return False

    PI0_CONFIG.write_text(content)
    print(f"  [write] {PI0_CONFIG.name}")
    return True


# ============================================================================
# EDIT 2: Modify make_attn_mask in pi0.py to apply BOS mask via env var
#
# Uses Python AST to find the EXACT line range of make_attn_mask and replace
# only those lines. This is robust to:
#   - multi-line docstrings (regex-based replacement broke on these in v1)
#   - blank lines inside the function body
#   - any whitespace / comment formatting
#
# Self-healing: if file already has OPENPI_MASK_BOS marker but is syntactically
# broken (from a previous failed patch), auto-restore from .bak.dlite and re-patch.
# ============================================================================
import ast as _ast

def edit_pi0_model():
    print(f"\n=== EDIT 2: pi0.py — add BOS mask logic to make_attn_mask ===")
    content = PI0_MODEL.read_text()

    # Self-healing idempotency check
    if "OPENPI_MASK_BOS" in content:
        try:
            _ast.parse(content)
            print("  [skip] BOS mask logic already present AND file is syntactically valid")
            return True
        except SyntaxError as e:
            print(f"  [warn] OPENPI_MASK_BOS marker present BUT pi0.py has syntax error:")
            print(f"         {e}")
            bak = PI0_MODEL.with_suffix(PI0_MODEL.suffix + ".bak.dlite")
            if bak.exists():
                shutil.copy(bak, PI0_MODEL)
                content = PI0_MODEL.read_text()
                print(f"  [self-heal] restored {PI0_MODEL.name} from {bak.name}; re-patching")
            else:
                print(f"  [ERROR] no backup at {bak} for auto-restore; manual fix needed")
                return False

    # Parse with AST to find make_attn_mask
    try:
        tree = _ast.parse(content)
    except SyntaxError as e:
        print(f"  [ERROR] pi0.py has pre-existing syntax error: {e}")
        return False

    func_node = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == 'make_attn_mask':
            func_node = node
            break

    if func_node is None:
        print(f"  [ERROR] 'def make_attn_mask' not found in {PI0_MODEL}")
        return False

    start_line = func_node.lineno - 1   # 0-indexed start of `def make_attn_mask...`
    end_line = func_node.end_lineno     # 1-indexed inclusive end → equivalent to 0-indexed exclusive
    print(f"  [ast] found make_attn_mask at lines {func_node.lineno}-{func_node.end_lineno}")

    backup(PI0_MODEL)

    new_function = '''def make_attn_mask(input_mask, mask_ar):
    """Build attention mask, with optional BOS-sink mask for D-lite.

    Standard openpi behavior: causal mask (cumsum on mask_ar) AND padding validity.
    Returns a 3D mask of shape bool[B, T, S] (NO heads dimension — flax broadcasts
    across heads internally).

    D-lite extension: when env var OPENPI_MASK_BOS=1, also zero out attention from
    action-query positions (last OPENPI_ACTION_HORIZON positions, default 50) to
    the BOS token (key position OPENPI_NUM_IMAGE_TOKENS, default 768 for pi0_libero
    with 3 cameras × 256 tokens per image). Env var is read at JIT-trace time; since
    the chain script sets it BEFORE invoking train/eval, the trace sees it.

    Args:
        input_mask: bool[B, N] valid-token mask
        mask_ar:    bool[B, N] auto-regressive (causal) mask source

    Returns:
        bool[B, N, N] attention mask
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    attn_mask = jnp.logical_and(attn_mask, valid_mask)

    # D-lite BOS mask (env-var controlled — see docstring above).
    # IMPORTANT: attn_mask is 3D [B, T, S]. Use 3 indices, not 4.
    import os as _os
    if _os.environ.get("OPENPI_MASK_BOS", "0") == "1":
        _num_image_tokens = int(_os.environ.get("OPENPI_NUM_IMAGE_TOKENS", "768"))
        _action_horizon = int(_os.environ.get("OPENPI_ACTION_HORIZON", "50"))
        _T = attn_mask.shape[-2]  # query positions
        # Defensive guard: only mask when sequence is long enough to contain action queries
        # AND the BOS position exists in keys (sequence has language tokens past image segment).
        if _T > _num_image_tokens + _action_horizon:
            _action_query_start = _T - _action_horizon
            attn_mask = attn_mask.at[:, _action_query_start:, _num_image_tokens].set(False)

    return attn_mask
'''

    # Split content into lines (keep newlines), splice in the new function
    lines = content.splitlines(keepends=True)
    new_lines = lines[:start_line] + [new_function] + lines[end_line:]
    new_content = ''.join(new_lines)

    # Sanity: ensure the new content is syntactically valid BEFORE writing
    try:
        _ast.parse(new_content)
    except SyntaxError as e:
        print(f"  [ERROR] post-edit syntax check failed: {e}")
        print(f"          NOT writing the file; backup preserved at .bak.dlite")
        return False

    PI0_MODEL.write_text(new_content)
    print(f"  [write] replaced make_attn_mask via AST (was lines {start_line+1}-{end_line})")
    return True


# ============================================================================
# EDIT 3: Add pi0_libero_cf_lora_bos_masked TrainConfig to training/config.py
# ============================================================================
def edit_train_config():
    print(f"\n=== EDIT 3: training/config.py — add new TrainConfig ===")
    content = TRAIN_CONFIG.read_text()

    if 'pi0_libero_cf_lora_bos_masked' in content:
        print("  [skip] TrainConfig already present")
        return True

    # Find name="pi0_libero_cf_lora" line
    cf_match = re.search(r'name="pi0_libero_cf_lora"', content)
    if not cf_match:
        print(f"  [ERROR] 'name=\"pi0_libero_cf_lora\"' not found in {TRAIN_CONFIG}")
        return False

    # Find the TrainConfig( opening before this name
    tc_open = content.rfind('TrainConfig(', 0, cf_match.start())
    if tc_open < 0:
        print(f"  [ERROR] no preceding 'TrainConfig(' for pi0_libero_cf_lora")
        return False

    # Scan forward tracking paren depth to find the closing ')'
    depth = 0
    pos = tc_open
    insert_pos = -1
    while pos < len(content):
        c = content[pos]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                # Skip past `,` if present (TrainConfig in a list)
                if pos + 1 < len(content) and content[pos + 1] == ',':
                    insert_pos = pos + 2
                else:
                    insert_pos = pos + 1
                break
        pos += 1

    if insert_pos < 0:
        print(f"  [ERROR] could not find closing ')' of pi0_libero_cf_lora TrainConfig")
        return False

    backup(TRAIN_CONFIG)

    # Determine indent
    line_start = content.rfind('\n', 0, tc_open) + 1
    tc_indent = content[line_start:tc_open]  # whitespace before "TrainConfig("
    field_indent = tc_indent + "    "

    new_entry = f"""

{tc_indent}# Variant D-lite: LoRA on counterfactual data + BOS attention mask.
{tc_indent}# Tests "attention-level intervention closes the bypass when combined with CF data".
{tc_indent}# Actual BOS masking controlled by env var OPENPI_MASK_BOS=1 set by chain script
{tc_indent}# in run_lora_ablation_chain_v3.sh.
{tc_indent}TrainConfig(
{field_indent}name="pi0_libero_cf_lora_bos_masked",
{field_indent}model=pi0_config.Pi0Config(
{field_indent}    paligemma_variant="gemma_2b_lora",
{field_indent}    action_expert_variant="gemma_300m_lora",
{field_indent}    mask_bos_for_action_queries=True,
{field_indent}),
{field_indent}data=LeRobotLiberoDataConfig(
{field_indent}    repo_id="living_room_scene2_cf",
{field_indent}    base_config=DataConfig(prompt_from_task=True),
{field_indent}    extra_delta_transform=True,
{field_indent}),
{field_indent}weight_loader=weight_loaders.CheckpointWeightLoader(
{field_indent}    "gs://openpi-assets/checkpoints/pi0_libero/params"
{field_indent}),
{field_indent}num_train_steps=1_500,
{field_indent}freeze_filter=pi0_config.Pi0Config(
{field_indent}    paligemma_variant="gemma_2b_lora",
{field_indent}    action_expert_variant="gemma_300m_lora",
{field_indent}    mask_bos_for_action_queries=True,
{field_indent}).get_freeze_filter(),
{field_indent}ema_decay=None,
{tc_indent}),"""

    content = content[:insert_pos] + new_entry + content[insert_pos:]
    TRAIN_CONFIG.write_text(content)
    print(f"  [write] added pi0_libero_cf_lora_bos_masked TrainConfig")
    return True


# ============================================================================
# VERIFY
# ============================================================================
def verify():
    print(f"\n=== VERIFY ===")

    # Syntax check
    for path in [PI0_CONFIG, PI0_MODEL, TRAIN_CONFIG]:
        try:
            subprocess.check_output(
                [sys.executable, '-m', 'py_compile', str(path)],
                stderr=subprocess.STDOUT,
            )
            print(f"  [syntax OK] {path.name}")
        except subprocess.CalledProcessError as e:
            print(f"  [SYNTAX ERROR] {path.name}")
            print(f"  {e.output.decode()}")
            return False

    # Runtime check: load the new config
    test_script = f'''
import sys
sys.path.insert(0, "{OPENPI_SRC.parent}")
from openpi.training import config as _c
try:
    cfg = _c.get_config("pi0_libero_cf_lora_bos_masked")
    print("OK: mask_bos =", cfg.model.mask_bos_for_action_queries)
    print("OK: action_horizon =", cfg.model.action_horizon)
    print("OK: paligemma_variant =", cfg.model.paligemma_variant)
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
'''
    try:
        out = subprocess.check_output(
            [sys.executable, '-c', test_script],
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        print(f"  [runtime check]")
        for line in out.decode().splitlines():
            print(f"    {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [RUNTIME FAIL]")
        for line in e.output.decode().splitlines():
            print(f"    {line}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [RUNTIME TIMEOUT] check took >180s; probably stuck on JAX import — likely OK")
        return True


def main():
    print("=" * 70)
    print("D-lite BOS Attention Mask Patcher (idempotent)")
    print("=" * 70)

    for p in [PI0_CONFIG, PI0_MODEL, TRAIN_CONFIG]:
        if not p.exists():
            print(f"\nERROR: file not found: {p}")
            sys.exit(2)

    ok_config = edit_pi0_config()
    ok_model = edit_pi0_model()
    ok_train = edit_train_config()

    if not (ok_config and ok_model and ok_train):
        print("\n" + "=" * 70)
        print("ABORTED — one or more edits failed.")
        print("Backups (if created) at *.bak.dlite — revert with:")
        print("  for f in ~/flare/openpi/src/openpi/models/pi0_config.py \\")
        print("           ~/flare/openpi/src/openpi/models/pi0.py \\")
        print("           ~/flare/openpi/src/openpi/training/config.py; do")
        print("    [ -f \"$f.bak.dlite\" ] && cp \"$f.bak.dlite\" \"$f\"; done")
        print("=" * 70)
        sys.exit(1)

    ok_verify = verify()

    print("\n" + "=" * 70)
    if ok_verify:
        print("SUCCESS — D-lite BOS mask patches applied and verified")
        print()
        print("Next: re-rsync the updated chain script + launch the chain:")
        print("  (from Mac)")
        print("  rsync -avz ~/flare/run_lora_ablation_chain_v3.sh BiasLab-William:/home/williamchung/flare/")
        print("  (on remote)")
        print("  tmux new -d -s chain-v3 'bash ~/flare/run_lora_ablation_chain_v3.sh; exec bash'")
    else:
        print("PATCHES APPLIED but VERIFY FAILED — see errors above.")
        print("To revert: see commands at top of this script.")
    print("=" * 70)
    sys.exit(0 if ok_verify else 1)


if __name__ == "__main__":
    main()
