#!/usr/bin/env bash
# run_final.sh — FINAL wrap-up runs, priority-ordered for the blog deadline.
#
# Fire ONCE in tmux and walk away:
#   tmux new -s final
#   bash ~/flare/run_final.sh 2>&1 | tee ~/flare/results/final_master.log
#   # detach: Ctrl+B then D     reattach: tmux a -t final
#
# Priority order (decisive-cheap-first; every stage survives a failure):
#   STAGE 1  steering L16, NATURAL alphas (+-20..150)  (~1-2h)  <- THE data-vs-capability verdict
#   STAGE 2  steering L14, natural alphas              (~1-2h)  <- second locus
#   STAGE 3  probe v2: commanded-target decode,        (~30-45m)<- replaces AUC=1.000; answers
#            GroupKFold + perm null + negation transfer,         backbone-vs-head + frozen-backbone
#            backbone AND expert loci
#   STAGE 4  placebo steering L16 (random direction)   (~1-2h)  <- direction-specificity control
#   STAGE 5  steering L12, natural alphas              (~1-2h)  <- completeness
#   STAGE 6  pi0.5 N=15 sweep IF checkpoint present    (~2.5h)  <- tighter scale-persistence CI
#   STAGE 7  summaries
#
# BEFORE: nvidia-smi  (if memory.free ~1GB someone else is on it — wait, ONE job at a time)
set -uo pipefail   # NO -e: continue past any single failure

PY=~/flare/openpi/.venv/bin/python
cd ~/flare
LOG=~/flare/results; mkdir -p "$LOG"
CHK=~/flare/checkpoints/pi0_libero_pt
PI05=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero
ts(){ date +"%Y-%m-%d %H:%M:%S"; }

echo "================================================================"
echo "FLARE final runs — started $(ts)"
echo "================================================================"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
ls -d "$CHK" >/dev/null 2>&1 || echo "!! WARNING: pi0 checkpoint not found at $CHK"

# --- STAGE 1+2: steering at NATURAL scale (defaults now -150..150 after the patch) ---
for L in 16 14; do
  echo ""; echo "[$(ts)] STAGE steer-L$L — contrastive direction, natural alphas"
  TORCH_COMPILE_DISABLE=1 $PY eval_steering_bank.py --checkpoint "$CHK" --config-name pi0_libero --layer $L --rollout-trials 8 --out-dir "$LOG/steering_bank_L${L}_fine" 2>&1 | tee "$LOG/final_steer_L$L.log" || echo "[steer L$L] FAILED — continuing"
done

# --- STAGE 3: corrected decodability probe (backbone + expert loci) ---
echo ""; echo "[$(ts)] STAGE probe-v2 — commanded-target decode, GroupKFold + perm null + negation transfer"
TORCH_COMPILE_DISABLE=1 $PY probe_decode_v2.py --checkpoint-dir "$CHK" --n-inits 10 --n-perms 200 --out-dir "$LOG/probe_decode_v2" 2>&1 | tee "$LOG/final_probe_v2.log" || echo "[probe v2] FAILED — continuing"

# --- STAGE 4: placebo control (random unit direction, same alphas, L16) ---
echo ""; echo "[$(ts)] STAGE placebo-L16 — random direction control"
TORCH_COMPILE_DISABLE=1 $PY eval_steering_bank.py --checkpoint "$CHK" --config-name pi0_libero --layer 16 --direction-mode random --rollout-trials 8 --out-dir "$LOG/steering_bank_L16_placebo" 2>&1 | tee "$LOG/final_steer_L16_placebo.log" || echo "[placebo] FAILED — continuing"

# --- STAGE 5: steering L12 (completeness) ---
echo ""; echo "[$(ts)] STAGE steer-L12 — contrastive direction, natural alphas"
TORCH_COMPILE_DISABLE=1 $PY eval_steering_bank.py --checkpoint "$CHK" --config-name pi0_libero --layer 12 --rollout-trials 8 --out-dir "$LOG/steering_bank_L12_fine" 2>&1 | tee "$LOG/final_steer_L12.log" || echo "[steer L12] FAILED — continuing"

# --- STAGE 6: pi0.5 N=15 sweep (only if the checkpoint re-downloaded) ---
if ls -d "$PI05" >/dev/null 2>&1; then
  BDDLS=$($PY -c "import json,os,analyze_prompt_sweep as a;d=os.path.expanduser('~/flare/custom_bddls/prompt_sweep');print(json.dumps([d+'/'+s+'.bddl' for s in a.PROMPT_CATEGORIES]))")
  echo ""; echo "[$(ts)] STAGE pi0.5 N=15 sweep"
  $PY remote_multimode_matrix.py --policy-config pi05_libero --checkpoint-dir "$PI05" --conditions 1 --n-trials 15 --task-ids 0 --out-suffix promptsweep_pi05_n15 --bddl-sweep-list "$BDDLS" 2>&1 | tee "$LOG/final_sweep_pi05.log" || echo "[pi0.5 sweep] FAILED — continuing"
  $PY analyze_prompt_sweep.py        --out-suffix promptsweep_pi05_n15 2>&1 | tee "$LOG/final_analyze_pi05.log" || true
  $PY analyze_prompt_sweep_strict.py --out-suffix promptsweep_pi05_n15 2>&1 | tee "$LOG/final_strict_pi05.log"  || true
else
  echo ""; echo "[$(ts)] STAGE pi0.5 SKIPPED — checkpoint not found at $PI05"
fi

# --- STAGE 7: summaries ---
echo ""; echo "[$(ts)] STAGE summaries"
for d in steering_bank_L16_fine steering_bank_L14_fine steering_bank_L16_placebo steering_bank_L12_fine; do
  f="$LOG/$d/steering_bank.json"
  if [ -f "$f" ]; then echo "--- $d ---"; $PY -c "import json;s=json.load(open('$f'));print(json.dumps(s.get('rollout',{}),indent=1))"; fi
done

echo ""; echo "================================================================"
echo "FLARE final runs — DONE $(ts).  Logs: $LOG/final_*.log"
echo "================================================================"
