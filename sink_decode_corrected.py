"""sink_decode_corrected.py — produce the CORRECTED sink-position interpretation.

The original sink_decode.py used `AutoTokenizer` with default padding (LEFT for
paligemma's tokenizer config), so it reported key 768 = `<pad>`. But the actual
pi0 preprocessing pipeline uses RIGHT padding (verified via check_pi0_tokenization
on remote — `tokenizer.padding_side` says 'left' but the in-pipeline embed_tokens
hook shows content at the start, padding at the end).

Under correct RIGHT-padding:
    lang_pos 0 = <bos> (token_id=2)  ← key 768 in the attention probe
    lang_pos 1..N = content tokens
    lang_pos N+1..47 = <pad>

This script reproduces the correct interpretation, runs entirely locally (just
needs paligemma tokenizer + numpy/json), and writes a corrected JSON.

Usage:
    python3 sink_decode_corrected.py \\
        --prompts "put both the alphabet soup and the tomato sauce in the basket" \\
                  "put the tomato sauce in the basket" \\
                  "stack the mugs" \\
        --sink-positions 768 303 816 837 769 770 776 \\
        --out ~/flare/results/sink_decode_T0_corrected
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


# ============================================================================
# Hardcoded tokenization fallback (from remote check_pi0_tokenization output)
# ============================================================================
# If paligemma tokenizer isn't available locally, fall back to known truth.
_KNOWN_TOKENIZATIONS = {
    "put both the alphabet soup and the tomato sauce in the basket": [
        (0, 2, "<bos>"), (1, 1065, "put"), (2, 2145, " both"),
        (3, 573, " the"), (4, 40027, " alphabet"), (5, 21078, " soup"),
        (6, 578, " and"), (7, 573, " the"), (8, 32493, " tomato"),
        (9, 16009, " sauce"), (10, 575, " in"), (11, 573, " the"),
        (12, 12220, " basket"), (13, 108, "\\n"),
        # 14-47 = <pad>
    ],
    "stack the mugs": [
        (0, 2, "<bos>"), (1, 8388, "stack"), (2, 573, " the"),
        (3, 95056, " mugs"), (4, 108, "\\n"),
        # 5-47 = <pad>
    ],
}


def tokenize_prompt(prompt: str, max_len: int = 48) -> List[Dict]:
    """Tokenize prompt with RIGHT padding (matching pi0's actual preprocessing)."""
    rows = []

    # Try the real paligemma tokenizer first
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        # Force right padding (matches what pi0 actually does)
        tokenizer.padding_side = "right"
        encoded = tokenizer(prompt, max_length=max_len,
                            padding="max_length", truncation=True,
                            return_tensors="pt")
        ids = encoded["input_ids"][0].tolist()
        for i, tid in enumerate(ids):
            text = tokenizer.decode([tid])
            rows.append({
                "lang_pos": i,
                "token_id": int(tid),
                "decoded": text,
                "is_padding": tid == 0,  # paligemma pad_token_id = 0
            })
        return rows
    except Exception as e:
        print(f"[tokenize] AutoTokenizer unavailable ({type(e).__name__}); "
              f"falling back to known tokenization data")

    # Fallback: use known tokenization from the remote check
    if prompt in _KNOWN_TOKENIZATIONS:
        known = _KNOWN_TOKENIZATIONS[prompt]
        for (i, tid, text) in known:
            rows.append({
                "lang_pos": i, "token_id": tid, "decoded": text,
                "is_padding": False,
            })
        # Pad the rest
        for i in range(len(known), max_len):
            rows.append({
                "lang_pos": i, "token_id": 0, "decoded": "<pad>",
                "is_padding": True,
            })
        return rows

    raise RuntimeError(
        f"Cannot tokenize prompt {prompt!r}: AutoTokenizer unavailable AND "
        f"prompt not in known_tokenizations dict. Add it manually or install "
        f"transformers + paligemma."
    )


def interpret_sink_position(key_pos: int, tokens: List[Dict],
                              image_end: int = 768, language_end: int = 816) -> str:
    """Map a key position to a human-readable interpretation under right-padding."""
    if 0 <= key_pos < image_end:
        per_image = image_end // 3
        img_idx = key_pos // per_image
        patch_in_image = key_pos % per_image
        side = int(per_image ** 0.5)
        row = patch_in_image // side
        col = patch_in_image % side
        return (f"IMAGE — image #{img_idx} (of 3), "
                f"patch ({row},{col}) of {side}×{side}")
    elif image_end <= key_pos < language_end:
        lang_pos = key_pos - image_end
        if lang_pos < len(tokens):
            t = tokens[lang_pos]
            pad_tag = " [PADDING]" if t["is_padding"] else ""
            return (f"LANGUAGE — lang_pos={lang_pos}, "
                    f"token_id={t['token_id']} {t['decoded']!r}{pad_tag}")
        return f"LANGUAGE — lang_pos={lang_pos}, out of range"
    elif language_end <= key_pos < 867:
        act_pos = key_pos - language_end
        return f"ACTION (self-attn) — action_pos={act_pos} (of 51 action-horizon tokens)"
    return f"OUT OF RANGE (max=867)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--sink-positions", nargs="+", type=int,
                   default=[768, 303, 816, 837, 769, 770, 776])
    p.add_argument("--max-token-len", type=int, default=48,
                   help="pi0_libero uses max_token_len=48")
    p.add_argument("--image-end", type=int, default=768,
                   help="Last image key position + 1 (3 images × 256 = 768)")
    p.add_argument("--language-end", type=int, default=816,
                   help="Last language key position + 1 (768 + 48 = 816)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize each prompt
    tokens_per_prompt = {}
    for prompt in args.prompts:
        tokens = tokenize_prompt(prompt, max_len=args.max_token_len)
        tokens_per_prompt[prompt] = tokens
        n_non_pad = sum(1 for t in tokens if not t["is_padding"])
        print(f"\n=== Prompt: {prompt!r} ===")
        print(f"  {n_non_pad} non-pad tokens; right-padded to {args.max_token_len}")
        print(f"  {'lang_pos':>8s}  {'key=768+pos':>10s}  {'tok_id':>6s}  {'pad?':>4s}  decoded")
        # Show only the content positions + first few pads
        show_until = min(n_non_pad + 3, args.max_token_len)
        for t in tokens[:show_until]:
            pad = "PAD" if t["is_padding"] else ""
            print(f"  {t['lang_pos']:>8d}  {args.image_end + t['lang_pos']:>10d}  "
                  f"{t['token_id']:>6d}  {pad:>4s}  {t['decoded']!r}")
        if show_until < args.max_token_len:
            print(f"  ... ({args.max_token_len - show_until} more positions, all <pad>)")

    # Interpret sink positions using the first prompt (others are similar)
    main_prompt = args.prompts[0]
    main_tokens = tokens_per_prompt[main_prompt]
    print(f"\n=== Sink-position interpretation (using prompt: {main_prompt[:40]!r}) ===")
    print(f"  Prefix layout: image [0:{args.image_end}] + "
          f"language [{args.image_end}:{args.language_end}] + actions [{args.language_end}:867]")
    interpretations = []
    for kp in args.sink_positions:
        interp = interpret_sink_position(kp, main_tokens, args.image_end, args.language_end)
        print(f"  key {kp:>4d}: {interp}")
        interpretations.append({"key_pos": kp, "interpretation": interp})

    # Save corrected JSON
    out = {
        "padding_side": "right (pi0's actual preprocessing — verified via "
                         "in-pipeline embed_tokens hook)",
        "image_end": args.image_end,
        "language_end": args.language_end,
        "max_token_len": args.max_token_len,
        "note": "Original sink_decode.py used default tokenizer LEFT padding "
                "which mis-labeled key 768 as <pad>. Under correct right-padding, "
                "key 768 = <bos> (content-free positional anchor, classic LLM "
                "BOS-sink pattern per Xiao et al. 2023).",
        "prompts": list(args.prompts),
        "tokens_per_prompt": {p: tokens_per_prompt[p] for p in args.prompts},
        "sink_positions": args.sink_positions,
        "interpretations": interpretations,
    }
    out_path = out_dir / "sink_decode_corrected.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] saved corrected JSON: {out_path}")


if __name__ == "__main__":
    main()
