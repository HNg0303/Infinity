"""
Diagnostic: is the teacher-forcing forward used by tools/optimize_ada_offset.py consistent with autoregressive_infer_cfg?
For one image, with the null text and NO offset:
  (a) teacher-forcing forward (the training path), eval mode, -> per-scale bit loss / acc
  (b) autoregressive_infer_cfg with gt_leak=all scales (every scale is fed the gt bits), logits captured -> per-scale bit loss / acc
If the training path is correct, (a) and (b) should match per scale (up to bf16 noise).
Also reports (a) for several null_len values.
"""
import os.path as osp
import sys
import argparse
from types import SimpleNamespace

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from tools.run_infinity import add_common_arguments, load_visual_tokenizer, load_transformer, transform
from tools.optimize_ada_offset import get_null_text_cond, get_scale_schedule


def per_scale_stats(logits_BLV, gt_BL, scale_schedule):
    B, L, _ = logits_BLV.shape
    lg = logits_BLV.float().reshape(B, L, -1, 2)
    loss = F.cross_entropy(lg.permute(0, 3, 1, 2), gt_BL, reduction='none').mean(-1)[0]  # L
    acc = (lg.argmax(-1) == gt_BL).float().mean(-1)[0]
    out, s = [], 0
    for pt, ph, pw in scale_schedule:
        n = pt * ph * pw
        out.append((loss[s:s + n].mean().item(), acc[s:s + n].mean().item() * 100))
        s += n
    return out, loss.mean().item(), acc.mean().item() * 100


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--null_lens', type=str, default='1,8,32,128')
    parser.set_defaults(cfg='1', h_div_w_template=-1.)
    args = parser.parse_args()
    device = torch.device('cuda')
    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.eval()
    print(f'[model] cfg_uncond {tuple(infinity.cfg_uncond.shape)}, use_flex_attn={infinity.use_flex_attn}, num_block_chunks={infinity.num_block_chunks}, '
          f'pad_to_multiplier={infinity.pad_to_multiplier}, training={infinity.training}')

    pil = Image.open(args.image).convert('RGB')
    _, scale_schedule, vae_scale_schedule = get_scale_schedule(args, pil.height / pil.width)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    x = transform(pil, tgt_h, tgt_w).unsqueeze(0).to(device)
    bsc = BitwiseSelfCorrection(vae, SimpleNamespace(noise_apply_layers=0, noise_apply_requant=1, noise_apply_strength=0.,
                                                     apply_spatial_patchify=0, debug_bsc=0))
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        raw = vae.encode_for_raw_features(x, scale_schedule=vae_scale_schedule)[0]
        x_BLC_wo_prefix, gt_ms_idx_Bl = bsc.flip_requant(vae_scale_schedule, None, raw, device)
    gt_BL = torch.cat(gt_ms_idx_Bl, 1).long()
    first_len = int(np.prod(scale_schedule[0]))
    x_BLC_wo_prefix = x_BLC_wo_prefix[:, :gt_BL.shape[1] - first_len]

    # (a) teacher forcing, several null lengths
    tf_stats = {}
    for nl in map(int, args.null_lens.split(',')):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = infinity(get_null_text_cond(infinity, 1, nl), x_BLC_wo_prefix.clone(), scale_schedule=scale_schedule)
        tf_stats[nl] = per_scale_stats(logits, gt_BL, scale_schedule)
        if nl == 1:
            tf_logits = logits.float()

    # (b) AR with every scale fed the gt bits, capture logits per scale
    captured = []
    orig = infinity.get_logits
    def hook(h, cond_BD):
        out = orig(h, cond_BD)
        captured.append(out.detach().float())
        return out
    infinity.get_logits = hook
    gt_ls = [g.reshape(1, 1, ph, pw, -1) for g, (_, ph, pw) in zip(gt_ms_idx_Bl, scale_schedule)]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        infinity.autoregressive_infer_cfg(
            vae=vae, scale_schedule=scale_schedule, label_B_or_BLT=get_null_text_cond(infinity, 1, 1), g_seed=0, B=1,
            cfg_list=[1.] * len(scale_schedule), tau_list=[1.] * len(scale_schedule), top_k=900, top_p=0.97, returns_vemb=1,
            cfg_insertion_layer=[0], vae_type=args.vae_type, ret_img=False, inference_mode=True,
            gt_leak=len(scale_schedule), gt_ls_Bl=gt_ls)
    infinity.get_logits = orig
    ar_logits = torch.cat(captured, 1)
    ar_stats = per_scale_stats(ar_logits, gt_BL, scale_schedule)

    print(f'\nper-scale bit loss / acc (null text, no offset); ln2 = {np.log(2):.4f}')
    hdr = 'scale  (h,w)    ' + '  '.join(f'TF null{nl:<4d}     ' for nl in tf_stats) + '  AR(gt-forced) null1   |TF1-AR| max'
    print(hdr)
    s = 0
    for si, (pt, ph, pw) in enumerate(scale_schedule):
        n = pt * ph * pw
        d = (tf_logits[:, s:s + n] - ar_logits[:, s:s + n]).abs().max().item()
        row = '  '.join(f'{tf_stats[nl][0][si][0]:.4f} {tf_stats[nl][0][si][1]:5.1f}%  ' for nl in tf_stats)
        print(f'{si:5d}  ({ph:2d},{pw:2d})  {row}  {ar_stats[0][si][0]:.4f} {ar_stats[0][si][1]:5.1f}%     {d:.3f}')
        s += n
    print('mean           ' + '  '.join(f'{tf_stats[nl][1]:.4f} {tf_stats[nl][2]:5.1f}%  ' for nl in tf_stats) + f'  {ar_stats[1]:.4f} {ar_stats[2]:5.1f}%')


if __name__ == '__main__':
    main()
