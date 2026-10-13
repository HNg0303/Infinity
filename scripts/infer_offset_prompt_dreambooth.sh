#!/bin/bash
# Quick offset -> prompt inference with the offsets of scripts/optimize_ada_offset_dreambooth_all.sh (default: dreambooth_ls0-6):
# the learned offset (null text) draws the scales it was trained on, then a separate pretrained text branch (T5 prompt, no
# offset, cfg) draws the remaining scales. Loads the models once for all subjects. See tools/infer_offset_prompt.py.
# usage: bash scripts/infer_offset_prompt_dreambooth.sh [subject ...]      (default: every subject in RUN_DIR)
#   env overrides: RUN_DIR=output/ada_offset/dreambooth_ls0-6 ITER=100 PROMPTS='a {} in the snow|...' ({} = class)
#                  OFFSET_SCALES=ckpt CFG=4 TAU=0.5 NUM_SAMPLES=4 SEED=0 BASELINE=1 OUT_DIR=... PROMPT_DIR_PREFIX= PYTHON=...
# output: ${OUT_DIR}/<subject>/${PROMPT_DIR_PREFIX}<prompt>/{seedN.jpg, baseline_seedN.jpg,
#         grid.jpg (rows: offset->prompt, pretrained baseline, train images; each described in a text panel on the left)}
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
RUN_DIR=${RUN_DIR:-output/ada_offset/dreambooth_ls0-6}
ITER=${ITER:-100}
PROMPTS=${PROMPTS:-'a {} in the jungle|a {} in the snow|a {} on the beach|a {} with a city in the background'}
OFFSET_SCALES=${OFFSET_SCALES:-ckpt}   # scales drawn by the offset branch; ckpt = the scales it was trained on
CFG=${CFG:-4}                         # cfg / tau of the prompt branch (scripts/infer.sh defaults)
TAU=${TAU:-0.5}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-0}
BASELINE=${BASELINE:-1}               # also sample the plain pretrained prompt with the same seeds
OUT_DIR=${OUT_DIR:-${RUN_DIR}/prompt_infer_it$(printf %05d "$ITER")}
PROMPT_DIR_PREFIX=${PROMPT_DIR_PREFIX:-}   # e.g. full_prompt_ -> <subject>/full_prompt_a_dog_on_a_beach/

model_path=weights/infinity_2b_reg.pth
vae_path=weights/infinity_vae_d32reg.pth
t5_path=weights/flan-t5-xl

[ -x "$PYTHON" ] || { echo "$PYTHON not found: run  bash scripts/setup_venv.sh  first"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "no such run: $RUN_DIR (run scripts/optimize_ada_offset_dreambooth_all.sh first)"; exit 1; }
# flan-t5-xl (~11 GB, safetensors only: torch<2.6 refuses .bin)
[ -f "$t5_path/model.safetensors.index.json" ] || $PYTHON -c "from huggingface_hub import snapshot_download as d; d('google/flan-t5-xl', local_dir='$t5_path', allow_patterns=['*.safetensors', '*.json', 'spiece.model'])"

subjects=$(IFS=,; echo "$*")
mkdir -p "$OUT_DIR"
$PYTHON tools/infer_offset_prompt.py \
    --pn 1M --model_type infinity_2b --model_path "$model_path" \
    --vae_type 32 --vae_path "$vae_path" \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1 \
    --text_channels 2048 --text_encoder_ckpt "$t5_path" --cfg "$CFG" --tau "$TAU" --seed "$SEED" \
    --run_dir "$RUN_DIR" --iter "$ITER" --subjects "$subjects" --prompts "$PROMPTS" --offset_scales "$OFFSET_SCALES" \
    --num_samples "$NUM_SAMPLES" --baseline "$BASELINE" --out_dir "$OUT_DIR" --prompt_dir_prefix "$PROMPT_DIR_PREFIX" \
    2>&1 | tee "${OUT_DIR}/infer.log"
echo "== done: ${OUT_DIR}"
