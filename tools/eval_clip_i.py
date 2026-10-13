"""
DreamBooth CLIP-I: cosine similarity between the CLIP image embeddings of a generated image and of each real (training)
image of its subject, averaged over the real images, then over seeds / subjects.

Expects the layout written by scripts/optimize_ada_offset_dreambooth_all.sh:
    <run_dir>/<subject>/itXXXXX/seed0.jpg .. seedN.jpg      (generated)
    <dataset>/<subject>/*.jpg                               (real images the offset was trained on)

    .venv/bin/python tools/eval_clip_i.py output/ada_offset/dreambooth_ls0-4 [output/ada_offset/dreambooth_ls0-6 ...]

Writes <run_dir>/clip_i.csv (one row per generated image) and <run_dir>/clip_i_summary.csv (subject x iteration, mean over seeds).
"""
import os
import os.path as osp
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(folder):
    return sorted(f for f in glob.glob(osp.join(folder, '*')) if f.lower().endswith(IMG_EXTS))


def numbered(folder, pattern):
    """entries of `folder` matching `pattern` (one int group), sorted by that int -> [(int, path)]"""
    out = []
    for name in os.listdir(folder):
        m = re.fullmatch(pattern, name)
        if m:
            out.append((int(m.group(1)), osp.join(folder, name)))
    return sorted(out)


@torch.no_grad()
def embed(paths, model, processor, device, batch_size):
    """L2-normalized CLIP image embeddings, (N, D) on cpu"""
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert('RGB') for p in paths[i:i + batch_size]]
        pixel_values = processor(images=imgs, return_tensors='pt')['pixel_values'].to(device)
        f = model.get_image_features(pixel_values=pixel_values).float()
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(feats, 0)


def evaluate_run(run_dir, args, model, processor, device, real_cache):
    rows = []
    for subject in sorted(os.listdir(run_dir)):
        sub_dir = osp.join(run_dir, subject)
        if not osp.isdir(sub_dir) or not numbered(sub_dir, r'it(\d+)'):
            continue
        real_dir = osp.join(args.dataset, subject)
        if subject not in real_cache:
            real_paths = list_images(real_dir)
            if not real_paths:
                print(f'[skip] {subject}: no real images in {real_dir}')
                continue
            real_cache[subject] = embed(real_paths, model, processor, device, args.batch_size)
        real = real_cache[subject]
        for it, it_dir in numbered(sub_dir, r'it(\d+)'):
            seeds = numbered(it_dir, r'seed(\d+)\.(?:jpg|jpeg|png)')
            if not seeds:
                continue
            gen = embed([p for _, p in seeds], model, processor, device, args.batch_size)
            sims = (gen @ real.T).mean(dim=1)  # per generated image: mean over the real images
            for (seed, _), s in zip(seeds, sims.tolist()):
                rows.append({'subject': subject, 'iter': it, 'seed': seed, 'clip_i': s, 'num_real': len(real)})
        print(f'[{subject}] ' + '  '.join(
            f'it{r_it}={np.mean([r["clip_i"] for r in rows if r["subject"] == subject and r["iter"] == r_it]):.4f}'
            for r_it in sorted({r['iter'] for r in rows if r['subject'] == subject})), flush=True)
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(run_dir, rows):
    """subject x iteration table (mean over seeds) + a MEAN row (mean over subjects)"""
    per = defaultdict(list)
    for r in rows:
        per[(r['subject'], r['iter'])].append(r['clip_i'])
    iters = sorted({it for _, it in per})
    subjects = sorted({s for s, _ in per})
    table = []
    for s in subjects:
        table.append({'subject': s, **{f'it{it}': f'{np.mean(per[(s, it)]):.4f}' if (s, it) in per else '' for it in iters}})
    table.append({'subject': 'MEAN', **{f'it{it}': f'{np.mean([np.mean(per[(s, it)]) for s in subjects if (s, it) in per]):.4f}' for it in iters}})
    write_csv(osp.join(run_dir, 'clip_i_summary.csv'), ['subject'] + [f'it{it}' for it in iters], table)

    print(f'\n== CLIP-I {run_dir}  ({len(subjects)} subjects, {len(rows)} generated images)')
    print(f'{"subject":<22}' + ''.join(f'{f"it{it}":>10}' for it in iters))
    for row in table:
        if row['subject'] == 'MEAN':
            print('-' * (22 + 10 * len(iters)))
        print(f'{row["subject"]:<22}' + ''.join(f'{row[f"it{it}"]:>10}' for it in iters))
    print(f'[saved] {osp.join(run_dir, "clip_i.csv")}, {osp.join(run_dir, "clip_i_summary.csv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dirs', nargs='+', help='output folder(s) of scripts/optimize_ada_offset_dreambooth_all.sh')
    parser.add_argument('--dataset', type=str, default='data/dreambooth/dataset', help='real images: <dataset>/<subject>/')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32', help='HF CLIP checkpoint, e.g. openai/clip-vit-large-patch14')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(args.model, use_safetensors=True).to(device).eval()  # torch<2.6 refuses .bin checkpoints
    processor = CLIPImageProcessor.from_pretrained(args.model)
    print(f'[clip] {args.model} on {device}')

    real_cache = {}  # subject -> real-image embeddings, shared across run dirs
    for run_dir in args.run_dirs:
        rows = evaluate_run(run_dir, args, model, processor, device, real_cache)
        if not rows:
            print(f'[skip] {run_dir}: no <subject>/itXXXXX/seed*.jpg found')
            continue
        write_csv(osp.join(run_dir, 'clip_i.csv'), ['subject', 'iter', 'seed', 'clip_i', 'num_real'], rows)
        summarize(run_dir, rows)
