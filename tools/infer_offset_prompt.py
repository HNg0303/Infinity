"""
Offset -> prompt sampling: the learned per-scale ada offset (on the null text, as trained) draws the offset scales, then a
separate pretrained text branch (T5 prompt, no offset, cfg against the pretrained null text) draws the remaining scales.

Batch rows of one sample:
    row 0  offset branch : cfg_uncond[:null_len] + ada offset              -> samples the offset scales (no cfg)
    row 1  cond branch   : T5(prompt), pretrained path, never sees the offset -> samples the later scales
    row 2  uncond branch : cfg_uncond[:len(prompt)], pretrained path          -> cfg partner of row 1
Every row is fed the same sampled tokens at every scale and keeps its own KV cache, so the prompt branch attends to the
offset-drawn coarse scales through its own pretrained activations. The model's own ada_offset machinery is not used.

    python tools/infer_offset_prompt.py --run_dir output/ada_offset/dreambooth_ls0-6 --iter 100 --subjects dog,cat \
        --pn 1M --model_path weights/infinity_2b_reg.pth --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
        --text_encoder_ckpt weights/flan-t5-xl --cfg 4 --tau 0.5 ...

--offset_scales defaults to the scales the offset was trained on; '' gives the plain pretrained prompt sample (baseline rows).
"""
import os
import os.path as osp
import re
import sys
import time
import argparse

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.run_infinity import add_common_arguments, load_visual_tokenizer, load_transformer, load_tokenizer, encode_prompt, transform
from tools.optimize_ada_offset import get_scale_schedule, parse_scales, reference_row

DEFAULT_PROMPTS = 'a {} in the jungle|a {} in the snow|a {} on the beach|a {} with a city in the background'


@torch.no_grad()
def offset_then_prompt_infer(infinity, vae, offset, null_len, prompt_cond, scale_schedule, offset_scales, seed, args):
    """offset: (S, 1, 6, C) ada offset (already multiplied by --offset_scale); returns one HW3 uint8 BGR image"""
    device = infinity.cfg_uncond.device
    p_kv, p_lens, _, _ = prompt_cond
    lens = [null_len, p_lens[0], p_lens[0]]
    kv_compact = torch.cat([infinity.cfg_uncond[:null_len].float(), p_kv.float(), infinity.cfg_uncond[:p_lens[0]].float()], 0)
    cu_seqlens_k = F.pad(torch.tensor(lens, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    rows = len(lens)
    infinity.rng.manual_seed(seed)
    rng = infinity.rng

    with torch.autocast('cuda', dtype=torch.bfloat16):
        kv_compact = infinity.text_norm(kv_compact)
        sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max(lens)))
        ca_kv = infinity.text_proj_for_ca(kv_compact), cu_seqlens_k, max(lens)
        last_stage = sos.unsqueeze(1).expand(rows, 1, -1) + infinity.pos_start.expand(rows, 1, -1)
        with torch.autocast('cuda', enabled=False):
            gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()  # rows, 1, 6, C
        offset = offset.to(gss)

        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            use_offset = si in offset_scales
            off_si = offset[0] if offset.shape[0] == 1 else offset[si] if si < offset.shape[0] else None  # shared or per-scale
            gss_si = torch.cat((gss[:1] + off_si, gss[1:])) if use_offset and off_si is not None else gss
            for block_idx, chunk in enumerate(infinity.block_chunks):
                if (infinity.add_lvl_embeding_only_first_block and block_idx == 0) or not infinity.add_lvl_embeding_only_first_block:
                    last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                for m in chunk.module:
                    last_stage = m(x=last_stage, cond_BD=gss_si, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=None,
                                   scale_schedule=scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)

            if use_offset:  # offset branch alone, as in training-time sampling
                logits_BlV = infinity.get_logits(last_stage[:1], cond_BD[:1]).mul(1 / args.offset_tau)
            else:           # pretrained prompt branch with cfg against the pretrained null text
                logits = infinity.get_logits(last_stage[1:], cond_BD[1:])
                logits_BlV = (args.cfg * logits[:1] + (1 - args.cfg) * logits[1:]).mul(1 / args.tau)

            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV.reshape(1, -1, 2), rng=rng, top_k=args.top_k, top_p=args.top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(1, pn[1], pn[2], -1).unsqueeze(1)  # 1, 1, h, w, d
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')  # 1, d, 1, h, w
            if si != len(scale_schedule) - 1:
                summed_codes += F.interpolate(codes, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up).squeeze(-3)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)  # 1, h*w, d
                last_stage = infinity.word_embed(infinity.norm0_ve(last_stage)).repeat(rows, 1, 1)
            else:
                summed_codes += codes
        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

        img = vae.decode(summed_codes.squeeze(-3))
    img = ((img + 1) / 2).permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    return img[0].cpu().numpy()  # H W 3, uint8, BGR


def read_classes(path):
    """prompts_and_classes.txt -> {subject: class}"""
    classes = {}
    for line in open(path):
        m = re.fullmatch(r'([\w]+),([\w ]+)', line.strip())
        if m and m.group(1) != 'subject_name':
            classes[m.group(1)] = m.group(2)
    return classes


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def fmt_scales(scales):
    """{0, 1, 2, 5} -> '0-2,5'"""
    out, s = [], sorted(scales)
    for i, x in enumerate(s):
        if i == 0 or x != s[i - 1] + 1:
            out.append([x, x])
        else:
            out[-1][1] = x
    return ','.join(f'{a}-{b}' if b > a else f'{a}' for a, b in out) or 'none'


def label_panel(text, h, w):
    """white (h, w) BGR panel with `text` word-wrapped and vertically centered; the first line is the title"""
    s = h / 1024
    title_font, font = ImageFont.load_default(size=int(52 * s)), ImageFont.load_default(size=int(40 * s))
    pad, gap = int(36 * s), int(14 * s)
    panel = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    lines = []  # (text, font)
    for pi, para in enumerate(text.split('\n')):
        f, line = (title_font if pi == 0 else font), ''
        for word in para.split(' '):
            cand = f'{line} {word}'.strip()
            if line and draw.textlength(cand, font=f) > w - 2 * pad:
                lines.append((line, f))
                line = word
            else:
                line = cand
        lines.append((line, f))
    heights = [f.size + gap for _, f in lines]
    y = max(pad, (h - sum(heights)) // 2)
    for (line, f), lh in zip(lines, heights):
        draw.text((pad, y), line, fill=(0, 0, 0) if f is title_font else (60, 60, 60), font=f)
        y += lh
    return np.array(panel)[:, :, ::-1]


def save_grid(path, rows, labels):
    """rows of HW3 BGR images -> one grid with a text panel describing each row on its left"""
    width = max(len(r) for r in rows)
    lines = []
    for row, label in zip(rows, labels):
        h, w = row[0].shape[:2]
        row = [label_panel(label, h, int(w * 0.75))] + list(row) + [np.full_like(row[0], 255)] * (width - len(row))
        lines.append(np.concatenate(row, axis=1))
    cv2.imwrite(path, np.concatenate(lines, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--run_dir', type=str, required=True, help='output of scripts/optimize_ada_offset_dreambooth_all.sh')
    parser.add_argument('--iter', type=int, default=100, help='which snapshot: <run_dir>/<subject>/itXXXXX/offset.pth')
    parser.add_argument('--subjects', type=str, default='', help='comma-separated; default: every subject in run_dir')
    parser.add_argument('--classes', type=str, default='data/dreambooth/dataset/prompts_and_classes.txt')
    parser.add_argument('--prompts', type=str, default=DEFAULT_PROMPTS, help="'|'-separated, '{}' is replaced by the subject's class")
    parser.add_argument('--offset_scales', type=str, default='ckpt', help="scales drawn by the offset branch: 'ckpt' (the trained loss_scales), '0-4', ...")
    parser.add_argument('--offset_scale', type=float, default=1.0, help='multiply the learned offset')
    parser.add_argument('--offset_tau', type=float, default=1.0, help='temperature on the offset scales (the offset was sampled with 1.0)')
    parser.add_argument('--baseline', type=int, default=1, choices=[0, 1], help='also sample the plain pretrained prompt (no offset) with the same seeds')
    parser.add_argument('--num_samples', type=int, default=4, help='seeds seed .. seed+n-1')
    parser.add_argument('--out_dir', type=str, default='', help='default: <run_dir>/prompt_infer_itXXXXX')
    parser.add_argument('--prompt_dir_prefix', type=str, default='', help='per-prompt folder: <out_dir>/<subject>/<prefix><prompt slug>/')
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.97)
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()
    args.cfg = float(args.cfg)
    assert args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify, 'only the bitwise, non-patchified setup (Infinity-2B) is supported'

    tag = f'it{args.iter:05d}'
    out_dir = args.out_dir or osp.join(args.run_dir, f'prompt_infer_{tag}')
    subjects = args.subjects.split(',') if args.subjects else sorted(
        s for s in os.listdir(args.run_dir) if osp.isfile(osp.join(args.run_dir, s, tag, 'offset.pth')))
    classes = read_classes(args.classes)
    templates = args.prompts.split('|')

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    prompt_cache = {}

    for subject in subjects:
        ckpt = torch.load(osp.join(args.run_dir, subject, tag, 'offset.pth'), map_location='cpu')
        offset = (ckpt['ada_offset'] * args.offset_scale).reshape(-1, 1, 6, infinity.C).cuda()
        h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else ckpt.get('h_div_w_template', 1.0)
        _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
        offset_scales = set(ckpt['loss_scales'] if args.offset_scales == 'ckpt' else parse_scales(args.offset_scales, len(scale_schedule)) if args.offset_scales else [])
        cls = classes.get(subject, subject.replace('_', ' '))
        sub_out = osp.join(out_dir, subject)
        os.makedirs(sub_out, exist_ok=True)
        print(f'[{subject}] class={cls!r}, offset {tag} x{args.offset_scale} on scales {sorted(offset_scales)}, prompt on the rest, cfg={args.cfg} tau={args.tau}')

        tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
        train_row = reference_row([transform(Image.open(p).convert('RGB'), tgt_h, tgt_w) for p in ckpt['images']])
        all_scales = set(range(len(scale_schedule)))
        seeds = f'seeds {args.seed}..{args.seed + args.num_samples - 1}'
        for template in templates:
            prompt = template.format(cls)
            if prompt not in prompt_cache:
                prompt_cache[prompt] = encode_prompt(text_tokenizer, text_encoder, prompt)
            prompt_dir = osp.join(sub_out, args.prompt_dir_prefix + slug(prompt))
            os.makedirs(prompt_dir, exist_ok=True)
            variants = [('', offset_scales, f'offset -> prompt\noffset ({tag}, trained on {fmt_scales(ckpt["loss_scales"])}, x{args.offset_scale}) '
                                            f'+ null text on scales {fmt_scales(offset_scales)}, tau={args.offset_tau}\n'
                                            f'prompt "{prompt}" on scales {fmt_scales(all_scales - offset_scales)}, cfg={args.cfg:g}, tau={args.tau}\n{seeds}')]
            if args.baseline:
                variants.append(('baseline_', set(), f'pretrained (no offset)\nprompt "{prompt}" on all scales, cfg={args.cfg:g}, tau={args.tau}\n{seeds}'))
            grid_rows, grid_labels = [], []
            for prefix, scales, label in variants:
                stt = time.time()
                row = [offset_then_prompt_infer(infinity, vae, offset, ckpt.get('null_len', 32), prompt_cache[prompt], scale_schedule, scales, args.seed + i, args)
                       for i in range(args.num_samples)]
                for i, img in enumerate(row):
                    cv2.imwrite(osp.join(prompt_dir, f'{prefix}seed{args.seed + i}.jpg'), img)
                grid_rows.append(row)
                grid_labels.append(label)
                print(f'  {label.splitlines()[0]} "{prompt}": {time.time() - stt:.1f}s')
            grid_rows.append(train_row)
            grid_labels.append(f'train images\n{len(train_row)} images of {subject}')
            save_grid(osp.join(prompt_dir, 'grid.jpg'), grid_rows, grid_labels)
            print(f'[saved] {prompt_dir}/grid.jpg')
