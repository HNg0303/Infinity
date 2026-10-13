#!/bin/bash
# Prompt injected into the learned ada offset (null text) through one conditioning path at a time, see tools/infer_offset_modspace.py:
#   mod delta : gss_null + offset + lambda*(gss_prompt - gss_null), ca_kv = null    (prompt only in the AdaLN modulation space)
#   ca prompt : gss_null + offset, ca_kv = T5(prompt)                               (prompt only through cross-attention)
#   mod prompt: gss_prompt + offset with sos / head adaLN on the prompt too, ca_kv = null
#   mod delta + ca prompt, plus reference rows: offset only, pretrained prompt (no offset), train images
# usage: bash scripts/infer_offset_modspace_dreambooth.sh [subject ...]      (default: every subject in RUN_DIR)
#   env overrides: RUN_DIR=output/ada_offset/dreambooth_ls0-6 ITER=100 PROMPTS=dreambooth|'a {} in the snow|...' ({} = class)
#                  DELTA_WEIGHTS=1 DELTA_SCALES=all OFFSET_SCALE=1 CFG=1[,2,3] TAU=1 NUM_SAMPLES=4 SEED=0
#                  VARIANTS=offset,mod_delta,mod_prompt,ca_prompt,combined,baseline SKIP_EXISTING=0 OUT_DIR=... PYTHON=...
# output: ${OUT_DIR}/<subject>/<prompt>/{offset_,<variant>_cfg<c>_,baseline_}seedN.jpg, prompt.txt, grid.jpg
#         score with: .venv/bin/python tools/eval_modspace.py ${OUT_DIR}
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
RUN_DIR=${RUN_DIR:-output/ada_offset/dreambooth_ls0-6}
ITER=${ITER:-100}
PROMPTS=${PROMPTS:-'a {} in the jungle|a {} in the snow|a {} on the beach|a {} with a city in the background'}  # dreambooth = the official 25 prompts per subject (live / object list)
DELTA_WEIGHTS=${DELTA_WEIGHTS:-1}     # comma-separated lambdas, one row each
DELTA_SCALES=${DELTA_SCALES:-all}     # scales where the modulation delta is added
OFFSET_SCALE=${OFFSET_SCALE:-1}       # multiply the learned offset (0 = control: no offset)
CFG=${CFG:-1}                         # cfg (comma-separated: one row each) / tau of the prompt-injected rows; cfg partner = offset only row
TAU=${TAU:-1}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-0}
VARIANTS=${VARIANTS:-offset,mod_delta,ca_prompt,combined,baseline}   # baseline = pretrained prompt (cfg 4, tau 0.5), same seeds
SKIP_EXISTING=${SKIP_EXISTING:-0}       # 1 = resume: keep the seed images already in OUT_DIR
OUT_DIR=${OUT_DIR:-${RUN_DIR}/modspace_infer_it$(printf %05d "$ITER")}

model_path=weights/infinity_2b_reg.pth
vae_path=weights/infinity_vae_d32reg.pth
t5_path=weights/flan-t5-xl

[ -x "$PYTHON" ] || { echo "$PYTHON not found: run  bash scripts/setup_venv.sh  first"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "no such run: $RUN_DIR (run scripts/optimize_ada_offset_dreambooth_all.sh first)"; exit 1; }

subjects=$(IFS=,; echo "$*")
mkdir -p "$OUT_DIR"
$PYTHON tools/infer_offset_modspace.py \
    --pn 1M --model_type infinity_2b --model_path "$model_path" \
    --vae_type 32 --vae_path "$vae_path" \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1 \
    --text_channels 2048 --text_encoder_ckpt "$t5_path" --cfg "$CFG" --tau "$TAU" --seed "$SEED" \
    --run_dir "$RUN_DIR" --iter "$ITER" --subjects "$subjects" --prompts "$PROMPTS" \
    --delta_weights "$DELTA_WEIGHTS" --delta_scales "$DELTA_SCALES" --offset_scale "$OFFSET_SCALE" \
    --num_samples "$NUM_SAMPLES" --variants "$VARIANTS" --skip_existing "$SKIP_EXISTING" --out_dir "$OUT_DIR" \
    2>&1 | tee -a "${OUT_DIR}/infer.log"
echo "== done: ${OUT_DIR}"
