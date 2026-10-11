#!/bin/bash
# Optimize Infinity-2B's per-scale adaLN offset on scales 0-4 for EVERY DreamBooth subject folder (one run per subject),
# keeping the offset after 50 and 100 steps and sampling 4 seeds (null text, cfg=1, offset added on scales 0-4) at each.
# usage: bash scripts/optimize_ada_offset_dreambooth_all.sh [subject ...]      (default: every folder in data/dreambooth/dataset/)
#   env overrides: PN=1M ITERS=100 SAVE_EVERY=50 LOSS_SCALES=0-4 OFFSET_MODE=per_scale BATCH_SIZE=2 GRAD_ACCUM=2 LR=5e-3
#                  NUM_SAMPLES=4 SEED=0 OUT_DIR=... PYTHON=... FORCE=1 (redo subjects that are already done)
# output:
#   ${OUT_DIR}/<subject>/optimize.log
#   ${OUT_DIR}/<subject>/it00050/{offset.pth, seed0.jpg .. seed3.jpg, infer_vs_train.jpg}   (top row: samples, bottom row: training images)
#   ${OUT_DIR}/<subject>/it00100/{...same...}
set -uo pipefail
cd "$(dirname "$0")/.."

dataset=data/dreambooth/dataset

PYTHON=${PYTHON:-.venv/bin/python}   # torch 2.5.1 env, created by scripts/setup_venv.sh
PN=${PN:-1M}
ITERS=${ITERS:-100}
SAVE_EVERY=${SAVE_EVERY:-50}        # offset + samples kept every N steps (and at the end)
LOSS_SCALES=${LOSS_SCALES:-0-4}     # scales the offset is trained on and applied on at inference
OFFSET_MODE=${OFFSET_MODE:-per_scale}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-5e-3}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-0}                     # samples use seeds SEED .. SEED+NUM_SAMPLES-1
FORCE=${FORCE:-0}
OUT_DIR=${OUT_DIR:-output/ada_offset/dreambooth_ls${LOSS_SCALES}}

model_path=weights/infinity_2b_reg.pth
vae_path=weights/infinity_vae_d32reg.pth

[ -x "$PYTHON" ] || { echo "$PYTHON not found: run  bash scripts/setup_venv.sh  first"; exit 1; }
(( ITERS % SAVE_EVERY == 0 )) || { echo "ITERS=$ITERS must be a multiple of SAVE_EVERY=$SAVE_EVERY"; exit 1; }

if [ $# -gt 0 ]; then
    subjects=("$@")
else
    mapfile -t subjects < <(find "$dataset" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

# weights (~10.4 GB, downloaded once)
for f in "$model_path" "$vae_path"; do
    [ -f "$f" ] || $PYTHON -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/Infinity', '$(basename "$f")', local_dir='weights')" || exit 1
done

common=(
    --pn "$PN" --model_type infinity_2b --model_path "$model_path"
    --vae_type 32 --vae_path "$vae_path"
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1
    --text_channels 2048 --seed "$SEED" --offset_infer_scales "$LOSS_SCALES"
)

# move one snapshot (offset + samples grid from optimize_ada_offset.py) into <subject>/itXXXXX/
collect() {  # collect <offset.pth> <grid stem, no .jpg> <dst dir>
    local offset=$1 stem=$2 dst=$3
    mkdir -p "$dst"
    mv "$offset" "$dst/offset.pth"
    for ((i = 0; i < NUM_SAMPLES; i++)); do
        mv "${stem}_r0_${i}.jpg" "$dst/seed$((SEED + i)).jpg"
    done
    mv "${stem}.jpg" "$dst/infer_vs_train.jpg"
}

mkdir -p "$OUT_DIR"
last=$(printf 'it%05d' "$ITERS")
failed=()
for subject in "${subjects[@]}"; do
    images=${dataset}/${subject}
    sub_out=${OUT_DIR}/${subject}
    raw=${sub_out}/_raw
    if [ ! -d "$images" ]; then
        echo "== skip ${subject}: no such folder ${images}"; failed+=("$subject"); continue
    fi
    if [ "$FORCE" != 1 ] && [ -f "${sub_out}/${last}/infer_vs_train.jpg" ]; then
        echo "== skip ${subject}: already done (${sub_out})"; continue
    fi
    rm -rf "$sub_out"; mkdir -p "$raw"

    echo "== [${subject}] optimizing ${OFFSET_MODE} ada offset on scales ${LOSS_SCALES}, $(ls "$images" | wc -l) images, ${ITERS} steps -> ${sub_out}"
    if ! $PYTHON tools/optimize_ada_offset.py --mode optimize "${common[@]}" \
        --images "$images" --hflip 1 --grad_checkpointing 1 --offset_mode "$OFFSET_MODE" \
        --iters "$ITERS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" \
        --loss_scales "$LOSS_SCALES" --log_every 10 --num_samples "$NUM_SAMPLES" \
        --sample_every "$SAVE_EVERY" --save_offset_every "$SAVE_EVERY" \
        --save_offset "${raw}/offset.pth" --save_file "${raw}/samples.jpg" \
        > "${sub_out}/optimize.log" 2>&1; then
        echo "   FAILED, see ${sub_out}/optimize.log"; tail -n 5 "${sub_out}/optimize.log"; failed+=("$subject"); continue
    fi
    grep -E '^\[it ' "${sub_out}/optimize.log" | tail -n 1

    # intermediate snapshots come from --save_offset_every / --sample_every, the last one from --save_offset / --save_file
    for ((it = SAVE_EVERY; it < ITERS; it += SAVE_EVERY)); do
        tag=$(printf 'it%05d' "$it")
        collect "${raw}/offset_${tag}.pth" "${raw}/samples_progress/${tag}" "${sub_out}/${tag}"
    done
    collect "${raw}/offset.pth" "${raw}/samples" "${sub_out}/${last}"
    rm -rf "$raw"
done

echo "== done: ${OUT_DIR} ($(( ${#subjects[@]} - ${#failed[@]} ))/${#subjects[@]} subjects ok)"
[ ${#failed[@]} -eq 0 ] || { echo "   failed: ${failed[*]}"; exit 1; }
