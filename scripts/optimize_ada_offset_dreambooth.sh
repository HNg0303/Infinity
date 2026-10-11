#!/bin/bash
# Optimize Infinity-2B's per-scale adaLN offset (each scale learns its own (6, C) offset) on ONE DreamBooth subject folder,
# then sample with the learned offset at several strengths.
# usage: bash scripts/optimize_ada_offset_dreambooth.sh [subject]      (default: dog; see data/dreambooth/dataset/)
#   env overrides: PN=1M ITERS=100 LOSS_SCALES=all OFFSET_MODE=per_scale BATCH_SIZE=2 GRAD_ACCUM=2 LR=5e-3 NUM_SAMPLES=4 SAMPLE_EVERY=10 OUT_DIR=... PYTHON=...
set -euo pipefail
cd "$(dirname "$0")/.."

subject=${1:-dog}
images=data/dreambooth/dataset/${subject}

PYTHON=${PYTHON:-.venv/bin/python}   # torch 2.5.1 env, created by scripts/setup_venv.sh
PN=${PN:-1M}
ITERS=${ITERS:-100}
LOSS_SCALES=${LOSS_SCALES:-all}     # per-scale offsets only get a gradient on scales in the loss
OFFSET_MODE=${OFFSET_MODE:-per_scale}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-5e-3}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}    # samples every N steps -> ${out_dir}/samples_vs_targets_progress/itXXXXX.jpg
out_dir=${OUT_DIR:-output/ada_offset/${subject}_ls${LOSS_SCALES}}   # e.g. dog_ls0-5, dog_lsall

model_path=weights/infinity_2b_reg.pth
vae_path=weights/infinity_vae_d32reg.pth

[ -x "$PYTHON" ] || { echo "$PYTHON not found: run  bash scripts/setup_venv.sh  first"; exit 1; }
[ -d "$images" ] || { echo "no such subject folder: $images"; ls data/dreambooth/dataset; exit 1; }

# weights (~10.4 GB, downloaded once)
for f in "$model_path" "$vae_path"; do
    [ -f "$f" ] || $PYTHON -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/Infinity', '$(basename "$f")', local_dir='weights')"
done

mkdir -p "$out_dir"
common=(
    --pn "$PN" --model_type infinity_2b --model_path "$model_path"
    --vae_type 32 --vae_path "$vae_path"
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1
    --text_channels 2048 --seed 0 --offset_infer_scales "$LOSS_SCALES"
)

echo "== optimizing ${OFFSET_MODE} ada offset on ${images} ($(ls "$images" | wc -l) images) -> ${out_dir}"
$PYTHON tools/optimize_ada_offset.py --mode optimize "${common[@]}" \
    --images "$images" --hflip 1 --grad_checkpointing 1 --offset_mode "$OFFSET_MODE" \
    --iters "$ITERS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" \
    --loss_scales "$LOSS_SCALES" --log_every 10 --num_samples "$NUM_SAMPLES" --sample_every "$SAMPLE_EVERY" \
    --save_offset "${out_dir}/offset.pth" --save_file "${out_dir}/samples_vs_targets.jpg" \
    2>&1 | tee "${out_dir}/optimize.log"

# offset strength sweep (null text, cfg=1); the offset is applied on the scales it was trained on
for s in 0.0 0.5 1.0 1.5; do
    $PYTHON tools/optimize_ada_offset.py --mode infer "${common[@]}" \
        --offset_path "${out_dir}/offset.pth" --offset_scale "$s" --num_samples "$NUM_SAMPLES" \
        --save_file "${out_dir}/infer_scale${s}.jpg" 2>&1 | tee -a "${out_dir}/infer.log" | grep -E '^\[(infer|saved)\]|Error' || true
done
echo "== done: ${out_dir}"
