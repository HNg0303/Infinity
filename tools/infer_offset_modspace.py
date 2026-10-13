"""
Inject the prompt into a learned ada offset sample (trained on the null text) through one conditioning path at a time:

    mod_delta    : gss = gss_null + offset + lambda * (gss_prompt - gss_null), ca_kv = null text
                   (the prompt only moves the shared adaLN modulation; at lambda=1 gss = gss_prompt + offset)
    mod_prompt   : gss = gss_prompt + offset, sos and head adaLN on the prompt,  ca_kv = null text
                   (the whole modulation path sees the prompt, only cross-attention stays null)
    ca_prompt    : gss = gss_null + offset,                                    ca_kv = T5(prompt)
                   (the prompt only enters through cross-attention)
    mod_delta+ca : both of the above

gss_* = shared_ada_lin(text_proj_for_sos(text)), the 6 modulation vectors shared by every block; gss_null uses the null text the
offset was trained on (cfg_uncond[:null_len]). In these rows the sos token and the head's adaLN (head_nm) always see the null
text (except in mod_prompt), so the prompt reaches the model only through the path named above. With --cfg != 1 the cfg partner is the plain offset
row (gss_null + offset, null ca_kv), so guidance amplifies only what the prompt adds, not the subject.
Reference rows: the plain offset sample (training-time sampling), the pretrained prompt sample (no offset) and the train images.

    python tools/infer_offset_modspace.py --run_dir output/ada_offset/dreambooth_ls0-6 --iter 100 --subjects dog,cat \
        --pn 1M --model_path weights/infinity_2b_reg.pth --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
        --text_encoder_ckpt weights/flan-t5-xl --delta_weights 1,2 --cfg 1 --tau 1 ...
"""
import os
import os.path as osp
import sys
import time
import argparse

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.run_infinity import add_common_arguments, load_visual_tokenizer, load_transformer, load_tokenizer, encode_prompt, transform
from tools.optimize_ada_offset import get_scale_schedule, parse_scales, reference_row
from tools.infer_offset_prompt import DEFAULT_PROMPTS, read_classes, slug, fmt_scales, save_grid

# DreamBooth (Ruiz et al. 2023) evaluation prompts, '{}' = class token (the unique token is dropped: the subject comes from the offset)
DREAMBOOTH_COMMON = ['a {} in the jungle', 'a {} in the snow', 'a {} on the beach', 'a {} on a cobblestone street', 'a {} on top of pink fabric',
                     'a {} on top of a wooden floor', 'a {} with a city in the background', 'a {} with a mountain in the background',
                     'a {} with a blue house in the background', 'a {} on top of a purple rug in a forest']
DREAMBOOTH_PROPERTY = ['a red {}', 'a purple {}', 'a shiny {}', 'a wet {}', 'a cube shaped {}']
DREAMBOOTH_OBJECT = DREAMBOOTH_COMMON + [
    'a {} with a wheat field in the background', 'a {} with a tree and autumn leaves in the background', 'a {} with the Eiffel Tower in the background',
    'a {} floating on top of water', 'a {} floating in an ocean of milk', 'a {} on top of green grass with sunflowers around it',
    'a {} on top of a mirror', 'a {} on top of the sidewalk in a crowded street', 'a {} on top of a dirt road', 'a {} on top of a white rug'] + DREAMBOOTH_PROPERTY
DREAMBOOTH_LIVE = DREAMBOOTH_COMMON + [
    'a {} wearing a red hat', 'a {} wearing a santa hat', 'a {} wearing a rainbow scarf', 'a {} wearing a black top hat and a monocle',
    'a {} in a chef outfit', 'a {} in a firefighter outfit', 'a {} in a police outfit', 'a {} wearing pink glasses',
    'a {} wearing a yellow shirt', 'a {} in a purple wizard outfit'] + DREAMBOOTH_PROPERTY
LIVE_SUBJECTS = {'cat', 'cat2', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8'}

NULL, PROMPT, NULL_P = 0, 1, 2  # text sources: offset's null text (cfg_uncond[:null_len]), T5(prompt), cfg_uncond[:len(prompt)]


@torch.no_grad()
def encode_texts(infinity, null_len, prompt_cond):
    """-> sos (3, D), gss (3, 1, 6, C), per-token ca features [(len_i, D)] for the NULL, PROMPT, NULL_P texts"""
    p_kv, p_lens, _, _ = prompt_cond
    lens = [null_len, p_lens[0], p_lens[0]]
    kv = torch.cat([infinity.cfg_uncond[:null_len].float(), p_kv.float(), infinity.cfg_uncond[:p_lens[0]].float()], 0)
    cu_seqlens_k = F.pad(torch.tensor(lens, device=kv.device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        kv = infinity.text_norm(kv)
        sos = infinity.text_proj_for_sos((kv, cu_seqlens_k, max(lens)))
        ca = infinity.text_proj_for_ca(kv)
    with torch.autocast('cuda', enabled=False):
        gss = infinity.shared_ada_lin(sos.float()).float().contiguous()
    return sos, gss, list(ca.split(lens))


@torch.no_grad()
def modspace_infer(infinity, vae, texts, offset, rows, scale_schedule, offset_scales, delta_scales, cfg, tau, seed, args):
    """
    rows: [cond] or [cond, uncond], each (cond_src, ca_src, use_offset, delta_w):
        sos / head adaLN / gss base from texts[cond_src], cross-attention on texts[ca_src],
        gss += offset[si] (if use_offset and si in offset_scales) + delta_w * (gss_prompt - gss_null) (if si in delta_scales)
    offset: (S, 1, 6, C); returns one HW3 uint8 BGR image
    """
    sos_all, gss_all, ca_all = texts
    n = len(rows)
    cond_src = [r[0] for r in rows]
    ca = [ca_all[r[1]] for r in rows]
    cu_seqlens_k = F.pad(torch.tensor([len(c) for c in ca], device=sos_all.device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    ca_kv = torch.cat(ca, 0), cu_seqlens_k, max(len(c) for c in ca)
    cond_BD = sos_all[cond_src]
    gss_base = gss_all[cond_src]
    delta = gss_all[PROMPT] - gss_all[NULL]
    offset = offset.to(gss_base)
    use_offset = torch.tensor([float(r[2]) for r in rows], device=gss_base.device).view(n, 1, 1, 1)
    delta_w = torch.tensor([float(r[3]) for r in rows], device=gss_base.device).view(n, 1, 1, 1)
    infinity.rng.manual_seed(seed)
    rng = infinity.rng

    with torch.autocast('cuda', dtype=torch.bfloat16):
        last_stage = cond_BD.unsqueeze(1) + infinity.pos_start.expand(n, 1, -1)
        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            gss_si = gss_base
            if si in offset_scales and si < offset.shape[0]:
                gss_si = gss_si + use_offset * offset[si]
            if si in delta_scales:
                gss_si = gss_si + delta_w * delta
            for block_idx, chunk in enumerate(infinity.block_chunks):
                if (infinity.add_lvl_embeding_only_first_block and block_idx == 0) or not infinity.add_lvl_embeding_only_first_block:
                    last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                for m in chunk.module:
                    last_stage = m(x=last_stage, cond_BD=gss_si, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=None,
                                   scale_schedule=scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)

            logits = infinity.get_logits(last_stage, cond_BD)
            logits_BlV = (cfg * logits[:1] + (1 - cfg) * logits[1:] if n == 2 else logits).mul(1 / tau)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV.reshape(1, -1, 2), rng=rng, top_k=args.top_k, top_p=args.top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(1, pn[1], pn[2], -1).unsqueeze(1)  # 1, 1, h, w, d
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')  # 1, d, 1, h, w
            if si != len(scale_schedule) - 1:
                summed_codes += F.interpolate(codes, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up).squeeze(-3)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)  # 1, h*w, d
                last_stage = infinity.word_embed(infinity.norm0_ve(last_stage)).repeat(n, 1, 1)
            else:
                summed_codes += codes
        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

        img = vae.decode(summed_codes.squeeze(-3))
    img = ((img + 1) / 2).permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    return img[0].cpu().numpy()  # H W 3, uint8, BGR


def build_variants(args, prompt, offset_desc, delta_scales, delta_weights, cfgs, all_scales):
    """-> [(file prefix, rows, cfg, tau, label)] for the variants in --variants; the prompt-injected rows get one entry per cfg"""
    seeds = f'seeds {args.seed}..{args.seed + args.num_samples - 1}'
    uncond = (NULL, NULL, True, 0.)
    delta_desc = f'on scales {fmt_scales(delta_scales)}' if delta_scales != all_scales else 'on all scales'
    variants = []

    def add(name, cond, title, desc):
        for cfg in cfgs:
            cfg_desc = f'cfg={cfg:g} vs offset row' if cfg != 1 else 'no cfg'
            variants.append((f'{name}_cfg{cfg:g}_', [cond, uncond] if cfg != 1 else [cond], cfg, args.tau,
                             f'{title}, {cfg_desc}\n{desc}; {offset_desc}; tau={args.tau:g}\n{seeds}'))

    if 'offset' in args.variants:
        variants.append(('offset_', [uncond], 1., args.offset_tau,
                         f'offset only (training-time sampling)\n{offset_desc}, null text everywhere, tau={args.offset_tau:g}\n{seeds}'))
    if 'mod_delta' in args.variants:
        for w in delta_weights:
            add(f'mod_delta{w:g}', (NULL, NULL, True, w), f'mod delta x{w:g} (ca_kv = null)',
                f'gss_null + offset + {w:g}*(gss_prompt - gss_null) {delta_desc}; "{prompt}"')
    if 'mod_prompt' in args.variants:
        add('mod_prompt', (PROMPT, NULL, True, 0.), 'prompt via sos + shared AdaLN + head AdaLN (ca_kv = null)',
            f'gss_prompt + offset, sos / head_nm on "{prompt}", ca_kv = null')
    if 'ca_prompt' in args.variants:
        add('ca_prompt', (NULL, PROMPT, True, 0.), 'prompt only via ca_kv', f'gss_null + offset, ca_kv = "{prompt}"')
    if 'combined' in args.variants:
        for w in delta_weights:
            add(f'mod_delta{w:g}_ca_prompt', (NULL, PROMPT, True, w), f'mod delta x{w:g} + prompt via ca_kv',
                f'gss_null + offset + {w:g}*(gss_prompt - gss_null) {delta_desc}, ca_kv = "{prompt}"')
    if 'baseline' in args.variants:
        variants.append(('baseline_', [(PROMPT, PROMPT, False, 0.), (NULL_P, NULL_P, False, 0.)], args.base_cfg, args.base_tau,
                         f'pretrained (no offset)\nprompt "{prompt}" on all paths, cfg={args.base_cfg:g}, tau={args.base_tau:g}\n{seeds}'))
    return variants


def prompt_templates(spec, subject):
    """--prompts: 'dreambooth' (the official 25 prompts: live list for dogs / cats, object list otherwise) or '|'-separated"""
    if spec == 'dreambooth':
        return DREAMBOOTH_LIVE if subject in LIVE_SUBJECTS else DREAMBOOTH_OBJECT
    return spec.split('|')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--run_dir', type=str, required=True, help='output of scripts/optimize_ada_offset_dreambooth_all.sh')
    parser.add_argument('--iter', type=int, default=100, help='which snapshot: <run_dir>/<subject>/itXXXXX/offset.pth')
    parser.add_argument('--subjects', type=str, default='', help='comma-separated; default: every subject in run_dir')
    parser.add_argument('--classes', type=str, default='data/dreambooth/dataset/prompts_and_classes.txt')
    parser.add_argument('--prompts', type=str, default=DEFAULT_PROMPTS, help="'dreambooth' (official 25 per subject) or '|'-separated; '{}' is replaced by the subject's class")
    parser.add_argument('--delta_weights', type=str, default='1', help="comma-separated lambdas for gss_null + offset + lambda*(gss_prompt - gss_null), one row each")
    parser.add_argument('--delta_scales', type=str, default='all', help="scales where the modulation delta is added: 'all', '0-6', ...")
    parser.add_argument('--offset_scales', type=str, default='all', help="scales where the offset is added ('all' = as trained: the offset is zero past its trained scales)")
    parser.add_argument('--offset_scale', type=float, default=1.0, help='multiply the learned offset')
    parser.add_argument('--offset_tau', type=float, default=1.0, help='temperature of the offset-only row (the offset was sampled with 1.0)')
    parser.add_argument('--variants', type=str, default='offset,mod_delta,ca_prompt,combined,baseline',
                        help='rows to sample, comma-separated subset of offset,mod_delta,mod_prompt,ca_prompt,combined,baseline')
    parser.add_argument('--base_cfg', type=float, default=4.0, help='cfg of the pretrained baseline row')
    parser.add_argument('--base_tau', type=float, default=0.5, help='tau of the pretrained baseline row')
    parser.add_argument('--num_samples', type=int, default=4, help='seeds seed .. seed+n-1')
    parser.add_argument('--out_dir', type=str, default='', help='default: <run_dir>/modspace_infer_itXXXXX')
    parser.add_argument('--skip_existing', type=int, default=0, choices=[0, 1], help='resume: reuse the seed images already in out_dir instead of sampling them')
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.97)
    parser.set_defaults(cfg='1', h_div_w_template=-1.)  # cfg (comma-separated: one row each) / tau: of the prompt-injected rows
    args = parser.parse_args()
    cfgs = [float(c) for c in args.cfg.split(',') if c]
    args.variants = set(args.variants.split(','))
    assert args.variants <= {'offset', 'mod_delta', 'mod_prompt', 'ca_prompt', 'combined', 'baseline'}, args.variants
    assert args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify, 'only the bitwise, non-patchified setup (Infinity-2B) is supported'
    delta_weights = [float(w) for w in args.delta_weights.split(',') if w]

    tag = f'it{args.iter:05d}'
    out_dir = args.out_dir or osp.join(args.run_dir, f'modspace_infer_{tag}')
    subjects = args.subjects.split(',') if args.subjects else sorted(
        s for s in os.listdir(args.run_dir) if osp.isfile(osp.join(args.run_dir, s, tag, 'offset.pth')))
    classes = read_classes(args.classes)

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
        null_len = ckpt.get('null_len', 32)
        h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else ckpt.get('h_div_w_template', 1.0)
        _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
        all_scales = set(range(len(scale_schedule)))
        offset_scales = set(parse_scales(args.offset_scales, len(scale_schedule)))
        delta_scales = set(parse_scales(args.delta_scales, len(scale_schedule)))
        trained = {si for si in range(offset.shape[0]) if offset[si].abs().sum() > 0} & offset_scales
        offset_desc = f'offset {tag} x{args.offset_scale:g} on scales {fmt_scales(trained)}'
        cls = classes.get(subject, subject.replace('_', ' '))
        sub_out = osp.join(out_dir, subject)
        os.makedirs(sub_out, exist_ok=True)
        print(f'[{subject}] class={cls!r}, {offset_desc}, delta weights {delta_weights} on scales {fmt_scales(delta_scales)}, cfg={args.cfg} tau={args.tau:g}')

        tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
        train_row = reference_row([transform(Image.open(p).convert('RGB'), tgt_h, tgt_w) for p in ckpt['images']])
        offset_row = None  # the offset-only row does not depend on the prompt: sampled once per subject
        for template in prompt_templates(args.prompts, subject):
            prompt = template.format(cls)
            if prompt not in prompt_cache:
                prompt_cache[prompt] = encode_prompt(text_tokenizer, text_encoder, prompt)
            texts = encode_texts(infinity, null_len, prompt_cache[prompt])
            delta_norm = (texts[1][PROMPT] - texts[1][NULL]).norm(dim=-1).flatten().tolist()
            off_norm = offset[sorted(trained)].norm(dim=-1).mean(0).flatten().tolist() if trained else [0.] * 6
            print(f'  "{prompt}": |delta| (g1 g2 s1 s2 b1 b2) = {" ".join(f"{x:.2f}" for x in delta_norm)}; '
                  f'mean |offset| = {" ".join(f"{x:.2f}" for x in off_norm)}')
            prompt_dir = osp.join(sub_out, slug(prompt))
            os.makedirs(prompt_dir, exist_ok=True)
            with open(osp.join(prompt_dir, 'prompt.txt'), 'w') as f:
                f.write(prompt + '\n')
            grid_rows, grid_labels = [], []
            for prefix, rows, cfg, tau, label in build_variants(args, prompt, offset_desc, delta_scales, delta_weights, cfgs, all_scales):
                stt = time.time()
                paths = [osp.join(prompt_dir, f'{prefix}seed{args.seed + i}.jpg') for i in range(args.num_samples)]
                if args.skip_existing and all(osp.isfile(p) for p in paths):
                    row = [cv2.imread(p) for p in paths]  # resume: reuse the images of an earlier run
                elif prefix == 'offset_' and offset_row is not None:
                    row = offset_row
                else:
                    row = [modspace_infer(infinity, vae, texts, offset, rows, scale_schedule, offset_scales, delta_scales, cfg, tau, args.seed + i, args)
                           for i in range(args.num_samples)]
                if prefix == 'offset_':
                    offset_row = row
                for p, img in zip(paths, row):
                    if not osp.isfile(p) or not args.skip_existing:
                        cv2.imwrite(p, img)
                grid_rows.append(row)
                grid_labels.append(label)
                print(f'    {label.splitlines()[0]}: {time.time() - stt:.1f}s')
            grid_rows.append(train_row)
            grid_labels.append(f'train images\n{len(train_row)} images of {subject}')
            save_grid(osp.join(prompt_dir, 'grid.jpg'), grid_rows, grid_labels)
            print(f'[saved] {prompt_dir}/grid.jpg')
