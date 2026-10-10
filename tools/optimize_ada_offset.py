"""
Optimize a learnable offset on Infinity's shared adaLN modulation (gamma1, gamma2, scale1, scale2, shift1, shift2)
so that the frozen model, conditioned on its own null text embedding (cfg_uncond -> text_norm -> text_proj_for_sos ->
shared_ada_lin, then + offset), reconstructs one target image or a set of images of one subject (DreamBooth-style).
By default every scale gets its own offset (S, 1, 6, C) (--offset_mode per_scale); --offset_mode shared uses one (1, 1, 6, C).
The offset is trained with the same bitwise cross-entropy loss as trainer.py (teacher forcing), optionally restricted to
some scales (--loss_scales), on randomly shuffled mini-batches of the training images.

Optimize (and save samples):
    python tools/optimize_ada_offset.py --mode optimize --images data/dog/ --pn 1M \
        --model_path weights/infinity_2b_reg.pth --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
        --batch_size 2 --grad_accum 2 --loss_scales 0-7 --save_offset out/offset.pth --save_file out/recon.jpg

Apply a learned offset at inference (several seeds / prompts, scaled by --offset_scale):
    python tools/optimize_ada_offset.py --mode infer --offset_path out/offset.pth --pn 1M ... --num_samples 4 \
        [--prompt "a dog on the beach|a dog in the snow" --text_encoder_ckpt weights/flan-t5-xl --cfg 3] --save_file out/test.jpg
"""
import os
import os.path as osp
import sys
import glob
import time
import argparse
from types import SimpleNamespace

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.models.layers import DropPath

from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity import add_common_arguments, load_visual_tokenizer, load_transformer, load_tokenizer, encode_prompt, transform

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def get_null_text_cond(infinity, B=1, null_len=1):
    """The model's own unconditional text input: cfg_uncond[:null_len] for every sample (what cond-drop / cfg feed in place of T5)."""
    device = infinity.cfg_uncond.device
    kv = infinity.cfg_uncond[:null_len].detach().float()
    kv_compact = kv.repeat(B, 1)
    lens = [null_len] * B
    cu_seqlens_k = torch.arange(0, (B + 1) * null_len, null_len, dtype=torch.int32, device=device)
    return kv_compact, lens, cu_seqlens_k, null_len


def get_scale_schedule(args, h_div_w):
    if args.h_div_w_template > 0:
        h_div_w = args.h_div_w_template
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    if args.apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule
    return float(h_div_w_template), scale_schedule, vae_scale_schedule


def parse_scales(spec, num_scales):
    """'all' | '0-5' | '0,1,2,8' | '0-3,10' -> sorted list of scale indices (0 = the 1x1 scale)."""
    if not spec or spec == 'all':
        return list(range(num_scales))
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    out = sorted(i for i in out if 0 <= i < num_scales)
    assert out, f'no valid scale in {spec!r} (model has {num_scales} scales)'
    return out


def list_images(spec):
    paths = []
    for p in spec.split(','):
        if osp.isdir(p):
            paths += sorted(f for f in glob.glob(osp.join(p, '*')) if f.lower().endswith(IMG_EXTS))
        elif p:
            paths.append(p)
    assert paths, f'no image found in {spec!r}'
    return paths


def bitwise_loss(logits_BLV, gt_BL, scale_schedule, loss_scales, label_smooth=0.0, reweight_loss_by_scale=False):
    """Same loss as InfinityTrainer.train_step (use_bit_label=1, bitloss_type='mean'), but only tokens of `loss_scales` count."""
    B, L, _ = logits_BLV.shape
    loss = nn.functional.cross_entropy(logits_BLV.reshape(B, L, -1, 2).permute(0, 3, 1, 2), gt_BL, label_smoothing=label_smooth, reduction='none')
    loss = loss.mean(dim=-1)  # B, L
    last_scale_area = np.sqrt(np.prod(scale_schedule[-1]))
    lw, per_scale = [], []
    for si, (pt, ph, pw) in enumerate(scale_schedule):
        n = pt * ph * pw
        w = (last_scale_area / np.sqrt(n) if reweight_loss_by_scale else 1.) if si in loss_scales else 0.
        lw.extend([w] * n)
    lw = torch.tensor(lw, device=loss.device, dtype=loss.dtype)[None, ...]
    lw = lw / lw.sum()
    mask = lw > 0
    bit_acc = (logits_BLV.reshape(B, L, -1, 2).argmax(dim=-1) == gt_BL).float().mean(dim=-1)[mask.expand(B, -1)].mean()
    return loss.mul(lw).sum(dim=-1).mean(), bit_acc


def freeze_for_offset_training(infinity, grad_checkpointing):
    """Freeze the backbone; use train() only so 'full-block' checkpointing is active, but keep drop-path / dropout off."""
    infinity.requires_grad_(False)
    infinity.cond_drop_rate = 0.
    if grad_checkpointing:
        infinity.train()
        for m in infinity.modules():
            if isinstance(m, (DropPath, nn.Dropout)):
                m.eval()
    else:
        infinity.eval()


def shuffled_batches(num_items, batch_size, seed):
    """Endless stream of mini-batches: each epoch is a fresh random permutation of the training images."""
    g = torch.Generator().manual_seed(seed)
    buf = []
    while True:
        while len(buf) < batch_size:
            buf += torch.randperm(num_items, generator=g).tolist()
        yield buf[:batch_size]
        buf = buf[batch_size:]


@torch.no_grad()
def generate(infinity, vae, text_cond_tuple, scale_schedule, args, seed, negative=None):
    infinity.eval()
    cfg_list = [args.cfg] * len(scale_schedule)
    tau_list = [args.tau] * len(scale_schedule)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, _, img_list = infinity.autoregressive_infer_cfg(
            vae=vae, scale_schedule=scale_schedule, label_B_or_BLT=text_cond_tuple, g_seed=seed,
            B=1, negative_label_B_or_BLT=negative, cfg_list=cfg_list, tau_list=tau_list,
            top_k=args.top_k, top_p=args.top_p, returns_vemb=1, cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type, ret_img=True, inference_mode=True, sampling_per_bits=args.sampling_per_bits,
        )
    return img_list[0].cpu().numpy()  # H W 3, uint8, BGR


def optimize(args, infinity, vae, device):
    paths = list_images(args.images)
    pils = [Image.open(p).convert('RGB') for p in paths]
    # one scale schedule for the whole set so images can be batched: template closest to the median aspect ratio
    h_div_w_template, scale_schedule, vae_scale_schedule = get_scale_schedule(args, float(np.median([p.height / p.width for p in pils])))
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    loss_scales = parse_scales(args.loss_scales, len(scale_schedule))
    fwd_scales = max(loss_scales) + 1  # later scales are not needed in the forward
    fwd_seq_len = int(np.array(scale_schedule)[:fwd_scales].prod(axis=1).sum())
    first_len = int(np.prod(scale_schedule[0]))
    # rope caches are keyed by full schedules; a truncated schedule reuses the full one (its first rows are the same positions)
    infinity.rope2d_freqs_grid.setdefault(str(tuple(scale_schedule[:fwd_scales])), infinity.rope2d_freqs_grid[str(tuple(scale_schedule))])
    print(f'[data] {len(paths)} image(s) -> {tgt_h}x{tgt_w}, h_div_w_template={h_div_w_template}, {len(scale_schedule)} scales, '
          f'loss on scales {loss_scales} (forward {fwd_scales} scales, {fwd_seq_len} tokens)')

    # encode every image (and its h-flip) once; teacher-forcing inputs / labels are re-derived per batch (BSC noise is random)
    inps = [transform(p, tgt_h, tgt_w) for p in pils]  # [-1, 1]
    if args.hflip:
        inps += [x.flip(-1) for x in inps]
    raw_features = []
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for x in inps:
            raw_features.append(vae.encode_for_raw_features(x.unsqueeze(0).to(device), scale_schedule=vae_scale_schedule)[0])
    raw_features = torch.cat(raw_features, 0)
    bsc = BitwiseSelfCorrection(vae, SimpleNamespace(
        noise_apply_layers=args.noise_apply_layers, noise_apply_requant=1, noise_apply_strength=args.noise_apply_strength,
        apply_spatial_patchify=args.apply_spatial_patchify, debug_bsc=0))

    freeze_for_offset_training(infinity, args.grad_checkpointing)
    num_offset_scales = len(scale_schedule) if args.offset_mode == 'per_scale' else 1
    offset = infinity.enable_ada_offset(cond_only=True, num_scales=num_offset_scales)  # after freezing, so it is the only trainable tensor
    optimizer = torch.optim.Adam([offset], lr=args.lr, weight_decay=args.weight_decay)
    print(f'[offset] trainable params: {offset.numel()} (shape {tuple(offset.shape)}), backbone frozen: {sum(p.requires_grad for p in infinity.parameters()) == 1}, '
          f'batch_size={args.batch_size} x grad_accum={args.grad_accum}')

    batches = shuffled_batches(len(raw_features), args.batch_size, args.seed)
    log = []
    for it in range(args.iters):
        stt = time.time()
        optimizer.zero_grad(set_to_none=True)
        it_loss, it_acc = 0., 0.
        for _ in range(args.grad_accum):
            idx = next(batches)
            with torch.no_grad():
                x_BLC_wo_prefix, gt_ms_idx_Bl = bsc.flip_requant(vae_scale_schedule, None, raw_features[idx], device)
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(fwd_seq_len - first_len), :]
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:, :fwd_seq_len].contiguous().long()

            text_cond_tuple = get_null_text_cond(infinity, B=len(idx), null_len=args.null_len)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits_BLV = infinity(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:fwd_scales])
            loss, bit_acc = bitwise_loss(logits_BLV.float(), gt_BL, scale_schedule[:fwd_scales], loss_scales, args.label_smooth, args.reweight_loss_by_scale)
            total = loss + args.reg_l2 * offset.pow(2).mean() if args.reg_l2 > 0 else loss
            (total / args.grad_accum).backward()
            it_loss += loss.item() / args.grad_accum
            it_acc += bit_acc.item() / args.grad_accum
        optimizer.step()

        log.append((it, it_loss, it_acc))
        if it % args.log_every == 0 or it == args.iters - 1:
            if offset.shape[0] == 1:  # norm per (gamma1, gamma2, scale1, scale2, shift1, shift2)
                norms, what = offset.detach().norm(dim=-1).flatten().tolist(), '|offset| (g1,g2,s1,s2,b1,b2)'
            else:                     # norm per scale
                norms, what = offset.detach().flatten(1).norm(dim=-1).tolist(), '|offset| per scale'
            print(f'[it {it:4d}/{args.iters}] loss={it_loss:.4f} bit_acc={it_acc * 100:.2f}% '
                  f'{what}=[{", ".join(f"{n:.3f}" for n in norms)}] {time.time() - stt:.2f}s/it', flush=True)

    if args.save_offset:
        os.makedirs(osp.dirname(osp.abspath(args.save_offset)), exist_ok=True)
        torch.save({
            'ada_offset': offset.detach().cpu(), 'offset_mode': args.offset_mode, 'null_len': args.null_len, 'pn': args.pn, 'h_div_w_template': h_div_w_template,
            'images': paths, 'loss_scales': loss_scales, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum,
            'iters': args.iters, 'lr': args.lr, 'log': log,
        }, args.save_offset)
        print(f'[saved] offset -> {args.save_offset}')

    if args.save_file:
        set_offset_infer_scales(infinity, args.offset_infer_scales, len(scale_schedule))
        samples = [generate(infinity, vae, get_null_text_cond(infinity, 1, args.null_len), scale_schedule, args, args.seed + i)
                   for i in range(args.num_samples)]
        targets = [((x.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).byte().numpy()[:, :, ::-1] for x in inps[:len(paths)]]
        save_rows(args.save_file, [samples, targets[:max(args.num_samples, 1)]])


def set_offset_infer_scales(infinity, spec, num_scales):
    infinity.ada_offset_infer_scales = None if spec in ('', 'all') else set(parse_scales(spec, num_scales))


def infer(args, infinity, vae, device):
    ckpt = torch.load(args.offset_path, map_location='cpu')
    infinity.enable_ada_offset(init=ckpt['ada_offset'] * args.offset_scale, cond_only=True)  # shared or per-scale, from the saved shape
    infinity.requires_grad_(False)
    null_len = ckpt.get('null_len', args.null_len)
    h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else ckpt.get('h_div_w_template', 1.0)
    _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
    set_offset_infer_scales(infinity, args.offset_infer_scales, len(scale_schedule))
    if args.prompt:
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        conds = [(p, encode_prompt(text_tokenizer, text_encoder, p)) for p in args.prompt.split('|')]
    else:
        conds = [('<null>', get_null_text_cond(infinity, 1, null_len))]
    rows = []
    for prompt, text_cond_tuple in conds:
        print(f'[infer] offset={args.offset_path} x{args.offset_scale} on scales {args.offset_infer_scales}, prompt={prompt!r}, cfg={args.cfg}')
        rows.append([generate(infinity, vae, text_cond_tuple, scale_schedule, args, args.seed + i) for i in range(args.num_samples)])
    save_rows(args.save_file, rows)


def save_rows(path, rows):
    """Save each image of each row, plus one grid (rows x columns) at `path`."""
    rows = [r for r in rows if len(r)]
    if not rows:
        return
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    stem, ext = osp.splitext(path)
    width = max(len(r) for r in rows)
    grid_rows = []
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            cv2.imwrite(f'{stem}_r{ri}_{ci}{ext}', img)
        pad = [np.zeros_like(row[0])] * (width - len(row))
        grid_rows.append(np.concatenate(list(row) + pad, axis=1))
    cv2.imwrite(path, np.concatenate(grid_rows, axis=0))
    print(f'[saved] {sum(len(r) for r in rows)} image(s) + grid -> {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--mode', type=str, default='optimize', choices=['optimize', 'infer'])
    parser.add_argument('--images', '--image', type=str, default='', help='optimize mode: an image, a folder, or a comma-separated list')
    parser.add_argument('--null_len', type=int, default=1, help='length of the null text cfg_uncond[:null_len]; 1 = the uncond branch of an empty prompt')
    parser.add_argument('--iters', type=int, default=500, help='optimizer steps')
    parser.add_argument('--batch_size', type=int, default=1, help='images per forward/backward, drawn from a shuffled permutation each epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='micro-batches accumulated per optimizer step (effective batch = batch_size * grad_accum)')
    parser.add_argument('--hflip', type=int, default=0, choices=[0, 1], help='also train on horizontally flipped copies')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--reg_l2', type=float, default=0., help='extra L2 penalty on the offset')
    parser.add_argument('--label_smooth', type=float, default=0.)
    parser.add_argument('--reweight_loss_by_scale', type=int, default=0, choices=[0, 1])
    parser.add_argument('--offset_mode', type=str, default='per_scale', choices=['per_scale', 'shared'], help='one offset per scale (S,1,6,C) or one for all scales (1,1,6,C)')
    parser.add_argument('--loss_scales', type=str, default='all', help="scales whose tokens are in the loss and backward, e.g. '0-5' (early scales), '0,2,4', 'all'")
    parser.add_argument('--offset_infer_scales', type=str, default='all', help="scales where the offset is applied at inference, same syntax as --loss_scales")
    parser.add_argument('--noise_apply_layers', type=int, default=0, help='bitwise self-correction noise on the first N scales (0 = pure teacher forcing)')
    parser.add_argument('--noise_apply_strength', type=float, default=0.3)
    parser.add_argument('--grad_checkpointing', type=int, default=1, choices=[0, 1])
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_offset', type=str, default='./ada_offset.pth')
    parser.add_argument('--offset_path', type=str, default='', help='learned offset (infer mode)')
    parser.add_argument('--offset_scale', type=float, default=1.0, help='multiply the learned offset at inference')
    parser.add_argument('--prompt', type=str, default='', help="infer mode: prompt(s) separated by '|' the offset is added on top of (default: null text)")
    parser.add_argument('--num_samples', type=int, default=4, help='samples per prompt (seeds seed .. seed+n-1)')
    parser.add_argument('--save_file', type=str, default='./ada_offset_samples.jpg')
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.97)
    parser.set_defaults(cfg='1', h_div_w_template=-1.)  # cfg=1: the null branch would equal the cond branch minus the offset
    args = parser.parse_args()
    args.cfg = float(args.cfg)

    device = torch.device('cuda')
    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    if args.mode == 'optimize':
        assert args.images, '--images is required in optimize mode'
        optimize(args, infinity, vae, device)
    else:
        assert args.offset_path, '--offset_path is required in infer mode'
        infer(args, infinity, vae, device)
