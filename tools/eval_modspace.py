"""
DreamBooth benchmark metrics (Ruiz et al. 2023) of the rows written by tools/infer_offset_modspace.py:
    DINO  : cosine between the DINO ViT-S/16 CLS embedding of a generated image and of each real image of its subject (mean over real images)
    CLIP-I: same with CLIP ViT-B/32 image embeddings
    CLIP-T: cosine between the CLIP image embedding of a generated image and the CLIP text embedding of its prompt
The 'real images' reference row is the mean pairwise DINO / CLIP-I between the real images of a subject (self-pairs excluded).

Expects <out_dir>/<subject>/<prompt slug>/{prompt.txt, <variant>_seedN.jpg}; the variant is the file prefix (offset, ca_prompt_cfg2, baseline, ...).

    .venv/bin/python tools/eval_modspace.py output/ada_offset/dreambooth_ls0-6/ca_edit_it00100 [...]

Writes <out_dir>/clip_scores.csv (one row per image) and <out_dir>/clip_scores_summary.csv
(variant x {DINO, CLIP-I, CLIP-T}: mean over the prompts and seeds of each subject, then over subjects).
"""
import os
import os.path as osp
import re
import argparse
from collections import defaultdict

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer

from eval_clip_i import list_images, embed, write_csv

METRICS = ('dino', 'clip_i', 'clip_t')
DINO_TRANSFORM = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # DINO's eval preprocessing


@torch.no_grad()
def embed_text(texts, model, tokenizer, device):
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    return torch.nn.functional.normalize(model.get_text_features(**tok).float(), dim=-1).cpu()


@torch.no_grad()
def embed_dino(paths, model, device, batch_size):
    """L2-normalized DINO CLS embeddings, (N, 384) on cpu"""
    feats = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([DINO_TRANSFORM(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]]).to(device)
        feats.append(torch.nn.functional.normalize(model(x).float(), dim=-1).cpu())
    return torch.cat(feats, 0)


def pairwise_mean(feats):
    """mean cosine over the off-diagonal pairs"""
    n = len(feats)
    sim = feats @ feats.T
    return ((sim.sum() - sim.diagonal().sum()) / (n * (n - 1))).item() if n > 1 else float('nan')


def variant_order(v):
    return (v != 'real images', v != 'offset', v == 'baseline', v)


def evaluate(out_dir, args, models, device, real_cache):
    clip, processor, tokenizer, dino = models
    rows, real_rows = [], []
    for subject in sorted(os.listdir(out_dir)):
        sub_dir = osp.join(out_dir, subject)
        prompt_dirs = sorted(d for d in os.listdir(sub_dir) if osp.isfile(osp.join(sub_dir, d, 'prompt.txt'))) if osp.isdir(sub_dir) else []
        if not prompt_dirs:
            continue
        if subject not in real_cache:
            real_paths = list_images(osp.join(args.dataset, subject))
            real_cache[subject] = (embed_dino(real_paths, dino, device, args.batch_size), embed(real_paths, clip, processor, device, args.batch_size))
        real_dino, real_clip = real_cache[subject]
        real_rows.append({'subject': subject, 'prompt': '', 'variant': 'real images', 'seed': -1,
                          'dino': pairwise_mean(real_dino), 'clip_i': pairwise_mean(real_clip), 'clip_t': float('nan')})
        for d in prompt_dirs:
            prompt = open(osp.join(sub_dir, d, 'prompt.txt')).read().strip()
            gen = []  # (variant, seed, path)
            for name in sorted(os.listdir(osp.join(sub_dir, d))):
                m = re.fullmatch(r'(.+)_seed(\d+)\.jpg', name)
                if m:
                    gen.append((m.group(1), int(m.group(2)), osp.join(sub_dir, d, name)))
            if not gen:
                continue
            paths = [p for _, _, p in gen]
            img = embed(paths, clip, processor, device, args.batch_size)
            dino_s = (embed_dino(paths, dino, device, args.batch_size) @ real_dino.T).mean(dim=1).tolist()
            clip_i = (img @ real_clip.T).mean(dim=1).tolist()
            clip_t = (img @ embed_text([prompt], clip, tokenizer, device).T)[:, 0].tolist()
            for (variant, seed, _), ds, ci, ct in zip(gen, dino_s, clip_i, clip_t):
                rows.append({'subject': subject, 'prompt': prompt, 'variant': variant, 'seed': seed, 'dino': ds, 'clip_i': ci, 'clip_t': ct})
        print(f'[{subject}] {len(prompt_dirs)} prompts', flush=True)
    return rows, real_rows


def summarize(out_dir, rows):
    """variant -> mean over subjects of (mean over that subject's prompts and seeds)"""
    per = defaultdict(lambda: defaultdict(list))  # variant -> subject -> [(dino, clip_i, clip_t)]
    for r in rows:
        per[r['variant']][r['subject']].append([r[k] for k in METRICS])
    variants = sorted(per, key=variant_order)
    subj_mean = {v: {s: np.mean(x, axis=0) for s, x in per[v].items()} for v in variants}
    table = []
    for v in variants:
        m = np.mean(list(subj_mean[v].values()), axis=0)
        table.append({'variant': v, **{k: f'{x:.4f}' for k, x in zip(METRICS, m)}, 'subjects': len(subj_mean[v]),
                      'images': sum(len(x) for x in per[v].values()) if v != 'real images' else 0})
    write_csv(osp.join(out_dir, 'clip_scores_summary.csv'), ['variant', *METRICS, 'subjects', 'images'], table)

    print(f'\n== {out_dir}')
    print(f'{"variant":<22}{"DINO":>8}{"CLIP-I":>8}{"CLIP-T":>8}{"subj":>6}{"imgs":>7}')
    for t in table:
        print(f'{t["variant"]:<22}' + ''.join(f'{t[k]:>8}' for k in METRICS) + f'{t["subjects"]:>6}{t["images"]:>7}')
    print(f'\nper subject, DINO / CLIP-I / CLIP-T')
    print(f'{"subject":<20}' + ''.join(f'{v[:20]:>22}' for v in variants))
    for s in sorted({r['subject'] for r in rows}):
        cells = [subj_mean[v].get(s) for v in variants]
        print(f'{s:<20}' + ''.join(f'{" / ".join(f"{x:.3f}" for x in c):>22}' if c is not None else f'{"":>22}' for c in cells))
    print(f'[saved] {osp.join(out_dir, "clip_scores.csv")}, {osp.join(out_dir, "clip_scores_summary.csv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dirs', nargs='+', help='output folder(s) of scripts/infer_offset_modspace_dreambooth.sh')
    parser.add_argument('--dataset', type=str, default='data/dreambooth/dataset', help='real images: <dataset>/<subject>/')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32', help='HF CLIP checkpoint (DreamBooth uses ViT-B/32)')
    parser.add_argument('--dino_model', type=str, default='vit_small_patch16_224.dino', help='timm DINO checkpoint (DreamBooth uses ViT-S/16)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip = CLIPModel.from_pretrained(args.model, use_safetensors=True).to(device).eval()  # torch<2.6 refuses .bin checkpoints
    processor = CLIPImageProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    dino = timm.create_model(args.dino_model, pretrained=True, num_classes=0).to(device).eval()  # CLS token features
    print(f'[models] {args.model}, {args.dino_model} on {device}')

    real_cache = {}
    for out_dir in args.out_dirs:
        rows, real_rows = evaluate(out_dir, args, (clip, processor, tokenizer, dino), device, real_cache)
        if not rows:
            print(f'[skip] {out_dir}: no <subject>/<prompt>/prompt.txt found')
            continue
        write_csv(osp.join(out_dir, 'clip_scores.csv'), ['subject', 'prompt', 'variant', 'seed', *METRICS], rows)
        summarize(out_dir, rows + real_rows)
