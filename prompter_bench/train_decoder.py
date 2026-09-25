"""
Adapt SAM's decoder to automatically generated prompts.

Phase 1 trained the HQ decoder with prompts made from the ground truth, so it trusts every prompt. Here the
prompter is frozen, its prompts are computed once, and the decoder's trainable HQ parts (hf_mlp_ar/tc,
compress_vit_feat, embedding_encoder, embedding_maskfeature) are fine-tuned so that the union of the prompted
masks matches the full ground truth -- the decoder can learn to drop false prompts and to correct shapes.
hf_token_ar/tc stay frozen: the encoder's token adapters read the same weights, so changing them would make the
cached encoder features stale.
"""
import os
import csv
import copy
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, sam_decode, union_logits,
                    upsample, make_prompts, binary_to_score, seed_everything, fold_of)
from build_cache import ENCODERS
from train import Batches, class_loss
from evaluate import kind_for
import prompters

TRAINABLE = ('hf_mlp_ar', 'hf_mlp_tc', 'compress_vit_feat', 'embedding_encoder', 'embedding_maskfeature')


@torch.no_grad()
def precompute_prompts(prompter, hard, data, kind, device):
    """Prompter output -> per image: {'own': {cls: bool mask}, 'prompts': {cls: sam_decode kwargs or None}}."""
    out = []
    for s in tqdm(range(0, len(data), 4), desc=f'prompts ({kind})', leave=False):
        pos = np.arange(s, min(s + 4, len(data)))
        batch = data.get(pos)
        o = prompter(batch)
        logits = F.interpolate(o['logits'].float(), size=batch['gt'].shape[-2:], mode='bilinear', align_corners=False)
        for b in range(len(pos)):
            item = {'own': {}, 'prompts': {}}
            for ch, (cls, label) in enumerate(CLASSES):
                m = (o['argmax'][b] == label) if hard else logits[b, ch] > 0
                score = binary_to_score(m) if hard else logits[b, ch]
                item['own'][cls] = m
                item['prompts'][cls] = make_prompts(m.cpu().numpy().astype(np.uint8), kind_for(kind, cls), score=score,
                                                    device=device)
            out.append(item)
    return out


def decode(climatesam, data, pos, cls, prompts):
    emb, interm0 = data.data['emb'][pos:pos + 1].float(), data.data['vit'][0][pos:pos + 1].float()
    return sam_decode(climatesam, emb, interm0, cls, **prompts)


@torch.no_grad()
def evaluate(climatesam, data, prompts, device, with_objects=False):
    climatesam.mask_decoder.eval()
    m, obj = SegMetrics(), ObjectMetrics()
    for i in range(len(data)):
        gt = data.data['gt'][i].long()
        preds = {}
        for cls, label in CLASSES:
            p = prompts[i]['prompts'][cls]
            lg = decode(climatesam, data, i, cls, p) if p else None
            preds[cls] = upsample(union_logits(lg).to(device)) > 0
            obj.update(preds[cls], gt == label, cls)
        m.update(preds['TC'], preds['AR'], gt)
    return {**m.compute(), **(obj.compute() if with_objects else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--prompter', required=True, help="'cgnet_finetuned' or a run name in results/runs/<encoder>/")
    ap.add_argument('--kind', default='bbox', choices=['bbox', 'hybrid', 'mask', 'bbox+mask'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--oof_prefix', default=None,
                    help="e.g. 'mpg_seg_fold': training prompts come from <prefix>{k}, the model that never saw fold k")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda')
    ckpt, mlp = ENCODERS[args.encoder]
    climatesam = load_climatesam(ckpt, mlp, device)

    if args.prompter.startswith('cgnet'):
        prompter, layers, hard = prompters.build(args.prompter)[0].to(device).eval(), [], True
    else:
        ck = torch.load(os.path.join(RESULTS, 'runs', args.encoder, args.prompter, 'best.pth'), map_location=device)
        prompter, layers = prompters.build(ck['arch'], climatesam=climatesam)
        prompter.load_state_dict(ck['state_dict'])
        prompter, hard = prompter.to(device).eval(), False
    layers = sorted(set(layers) | {0})

    splits = {}
    for split in ('train', 'val', 'test'):
        cache = FeatureCache(args.encoder, split)
        data = Batches(cache, layers, device)
        if hard:
            cg = torch.from_numpy(np.ascontiguousarray(cache.cgnet[cache.index])).to(device)
            base = data.get
            data.get = lambda pos, base=base, cg=cg: {**base(pos), 'cgnet': cg[np.sort(np.asarray(pos))]}
        if args.oof_prefix and split == 'train':
            folds = fold_of(len(cache))
            prompts = [None] * len(cache)
            for k in range(5):
                sub = copy.copy(cache)
                sub.index = cache.index[folds == k]
                fk = torch.load(os.path.join(RESULTS, 'runs', args.encoder, f'{args.oof_prefix}{k}', 'best.pth'), map_location=device)
                model_k, _ = prompters.build(fk['arch'], climatesam=climatesam)
                model_k.load_state_dict(fk['state_dict'])
                pk = precompute_prompts(model_k.to(device).eval(), False, Batches(sub, layers, device), args.kind, device)
                for j, pos in enumerate(np.where(folds == k)[0]):
                    prompts[pos] = pk[j]
        else:
            prompts = precompute_prompts(prompter, hard, data, args.kind, device)
        del data
        splits[split] = (Batches(cache, [0], device), prompts)  # the decoder only needs the embedding + block 0
    del prompter
    torch.cuda.empty_cache()

    name = f'decoder_adapt_{args.prompter}_{args.kind.replace("+", "_")}{"_oof" if args.oof_prefix else ""}_s{args.seed}'
    out_dir = os.path.join(RESULTS, 'runs', args.encoder, name)
    os.makedirs(out_dir, exist_ok=True)

    dec = climatesam.mask_decoder
    params = []
    for n, p in dec.named_parameters():
        p.requires_grad = n.split('.')[0] in TRAINABLE
        if p.requires_grad:
            params.append(p)
    n_params = sum(p.numel() for p in params)
    original = {k: v.clone() for k, v in dec.state_dict().items()}
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    train_data, train_prompts = splits['train']
    val_data, val_prompts = splits['val']
    rows = [{'epoch': 0, **{f'val_{k}': v for k, v in evaluate(climatesam, val_data, val_prompts, device).items()}}]
    best = dict(rows[0])
    best_state = {k: v.clone() for k, v in dec.state_dict().items()}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        dec.train()
        for mod_name, mod in dec.named_children():  # SAM's own decoder layers stay in eval mode
            if mod_name not in TRAINABLE:
                mod.eval()
        order = np.random.permutation(len(train_data))
        total, steps = 0.0, 0
        for s in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for i in order[s:s + 8]:
                gt = train_data.data['gt'][i].long()
                for cls, label in CLASSES:
                    p = train_prompts[i]['prompts'][cls]
                    if not p:
                        continue
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lg = union_logits(decode(climatesam, train_data, i, cls, p))
                    up = F.interpolate(lg[None, None].float(), size=gt.shape, mode='bilinear', align_corners=False)[0]
                    loss = loss + class_loss(up, (gt == label).float()[None], cls)
            if torch.is_tensor(loss):
                (loss / 8).backward()
                optimizer.step()
                total += float(loss) / 8
                steps += 1
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': total / max(steps, 1),
               **{f'val_{k}': v for k, v in evaluate(climatesam, val_data, val_prompts, device).items()}}
        rows.append(row)
        if row['val_Mean FG IoU'] > best['val_Mean FG IoU']:
            best = dict(row)
            best_state = {k: v.clone() for k, v in dec.state_dict().items()}
        print(f"[{name}] epoch {epoch} loss {row['train_loss']:.4f} val FG IoU {row['val_Mean FG IoU']:.4f}", flush=True)
        with open(os.path.join(out_dir, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows([{k: r.get(k, '') for k in rows[-1]} for r in rows])

    torch.save({k: v for k, v in best_state.items() if k.split('.')[0] in TRAINABLE}, os.path.join(out_dir, 'best_decoder.pth'))
    test_data, test_prompts = splits['test']
    own = SegMetrics()
    for i in range(len(test_data)):
        own.update(test_prompts[i]['own']['TC'], test_prompts[i]['own']['AR'], test_data.data['gt'][i].long())
    dec.load_state_dict(original)
    before = evaluate(climatesam, test_data, test_prompts, device, with_objects=True)
    dec.load_state_dict(best_state)
    after = evaluate(climatesam, test_data, test_prompts, device, with_objects=True)
    result = {'name': name, 'prompter': args.prompter, 'kind': args.kind, 'trainable_params': n_params,
              'best_epoch': best['epoch'], 'train_time_min': (time.time() - t_start) / 60,
              'test': {'prompter mask': own.compute(), 'SAM, Phase-1 decoder': before, 'SAM, adapted decoder': after}}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['test'], indent=2))


if __name__ == '__main__':
    main()
