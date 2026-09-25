"""
Builds the tables and figures of results/ from the raw outputs (results/eval, results/runs, CSVs of the studies).
Run after the experiments; safe to re-run at any time (only reads raw outputs, overwrites derived files).
"""
import os
import re
import csv
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import RESULTS

TABLES = os.path.join(RESULTS, 'tables')
FIGS = os.path.join(RESULTS, 'figures')
ENCODER_NAMES = {'infused_mlp1': 'Infused Token, MLP ratio 1.0', 'infused_mlp05': 'Infused Token, MLP ratio 0.5'}
METHODS = {  # key -> (display name, trainable parameters are filled in from the runs)
    'oracle_gt': 'Ground truth (oracle)',
    'cgnet_official': 'CG-Net (official weights)',
    'cgnet_finetuned': 'CG-Net (fine-tuned)',
    'logreg_l0_seg': 'Logistic regression, ViT block 1',
    'logreg_last_seg': 'Logistic regression, ViT block 12',
    'msf_seg': 'Multi-scale fusion',
    'msf_token_seg': 'Multi-scale fusion + token gate',
    'mpg_seg': 'Mask-prompt generator (segmentation loss)',
    'mpg_sam_e2e': 'Mask-prompt generator (end-to-end via SAM)',
    'mpg_twostage': 'Mask-prompt generator (two-stage)',
    'mpg_seg_smooth': 'Mask-prompt generator (segmentation loss, label smoothing)',
    'learned_prompt_k4': 'Learned static prompts (4 tokens / class)',
    'learned_prompt_k16': 'Learned static prompts (16 tokens / class)',
}
OUTPUTS = {
    'own': 'Prompter mask (no SAM)',
    'sam_bbox': 'SAM + box',
    'sam_bbox_e10': 'SAM + box (+10%)',
    'sam_point': 'SAM + points',
    'sam_bbox+point': 'SAM + box + points',
    'sam_mask': 'SAM + mask logits',
    'sam_bbox+mask': 'SAM + box + mask',
    'sam_hybrid': 'SAM + hybrid',
    'fused_hybrid': 'mean(prompter, SAM hybrid)',
    'sam_static': 'SAM + learned static prompts',
}
METRICS = ['TC IoU', 'AR IoU', 'BG IoU', 'Mean IoU', 'Mean FG IoU']
TC_COLOR, AR_COLOR = '#2ca02c', '#1f4fd1'


def method_key(name):
    return re.sub(r'_s\d+$', '', name)


# ------------------------------------------------------------
# LOADING
# ------------------------------------------------------------
def load_eval(encoder):
    rows, dec = [], []
    for f in sorted(glob.glob(os.path.join(RESULTS, 'eval', encoder, '*.json'))):
        name = os.path.basename(f)[:-5]
        d = json.load(open(f))
        seed = int(re.search(r'_s(\d+)$', name).group(1)) if re.search(r'_s(\d+)$', name) else 0
        for r in d['rows']:
            rows.append({**r, 'method': method_key(name), 'seed': seed})
        for cls, v in d['error_decomposition'].items():
            dec.append({'method': method_key(name), 'seed': seed, 'class': cls, **v})
    for f in sorted(glob.glob(os.path.join(RESULTS, 'runs', encoder, 'learned_prompt_*', 'summary.json'))):
        d = json.load(open(f))
        rows.append({'method': method_key(d['name']), 'seed': d['seed'], 'output': 'sam_static', **d['test']})
    return pd.DataFrame(rows), pd.DataFrame(dec)


def load_runs(encoder):
    summaries, logs = [], {}
    for d in sorted(glob.glob(os.path.join(RESULTS, 'runs', encoder, '*'))):
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, 'summary.json')) or name.startswith('decoder_adapt') or '_fold' in name:
            continue
        s = json.load(open(os.path.join(d, 'summary.json')))
        s['method'] = method_key(s['name'])
        summaries.append(s)
        logs[s['name']] = pd.read_csv(os.path.join(d, 'log.csv'))
    return pd.DataFrame(summaries), logs


def mean_std(df, by, cols):
    g = df.groupby(by)[cols]
    return g.mean(), g.std().fillna(0.0), g.size()


# ------------------------------------------------------------
# LATEX
# ------------------------------------------------------------
def fmt(m, s, n):
    return f'{m:.3f}' if n <= 1 else f'{m:.3f}$\\pm${s:.3f}'


def write_table(name, header, body, caption, label, colspec=None):
    os.makedirs(TABLES, exist_ok=True)
    colspec = colspec or 'l' + 'c' * (len(header) - 1)
    tex = lambda cells: ' & '.join(re.sub(r'(?<!\\)%', r'\\%', c) for c in cells) + ' \\\\'
    lines = ['\\begin{table}[htbp]', '\\centering', '\\small', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule',
             tex(header), '\\midrule']
    for row in body:
        lines.append('\\midrule' if row == 'MIDRULE' else tex(row))
    lines += ['\\bottomrule', '\\end{tabular}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\end{table}']
    with open(os.path.join(TABLES, f'{name}.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    plain = lambda c: re.sub(r'\$\\pm\$', ' ± ', re.sub(r'\$\\Delta\$', 'Δ', c)).replace('\\%', '%').replace('$', '').replace('\\_', '_')
    with open(os.path.join(TABLES, f'{name}.md'), 'w') as f:
        f.write('| ' + ' | '.join(plain(h) for h in header) + ' |\n|' + '---|' * len(header) + '\n')
        for row in body:
            if row != 'MIDRULE':
                f.write('| ' + ' | '.join(plain(c) for c in row) + ' |\n')
    with open(os.path.join(TABLES, f'{name}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([re.sub(r'\\[a-z]+|[{}$]', '', h) for h in header])
        for row in body:
            if row != 'MIDRULE':
                w.writerow([re.sub(r'\$\\pm\$', ' ± ', c) for c in row])


# ------------------------------------------------------------
# MAIN TABLES
# ------------------------------------------------------------
def main_tables(encoder, ev, dec, runs):
    if ev.empty:
        return
    tag = encoder
    params = runs.groupby('method')['trainable_params'].first().to_dict() if not runs.empty else {}
    params.update({'cgnet_official': 494232, 'cgnet_finetuned': 494232})
    order = [m for m in METHODS if m in set(ev['method'])]

    # Table A: own mask vs the best SAM conversion per method
    m, s, n = mean_std(ev, ['method', 'output'], METRICS)
    body = []
    for meth in order:
        first = True
        for out in ('own', 'sam_bbox', 'sam_hybrid', 'sam_static'):
            if (meth, out) not in m.index:
                continue
            key = (meth, out)
            body.append([METHODS[meth] if first else '', OUTPUTS[out]] +
                        [fmt(m.loc[key, c], s.loc[key, c], n.loc[key]) for c in METRICS])
            first = False
        body.append('MIDRULE')
    write_table(f'main_{tag}', ['Prompter', 'Output'] + METRICS, body[:-1],
                f'Automatic prompting on the ClimateNet test set (61 images), frozen ClimateSAM ({ENCODER_NAMES[encoder]}). '
                'Learned prompters: mean$\\pm$std over 3 seeds, checkpoint selected on 40 held-out training images.',
                f'tab:prompters_{tag}', 'llccccc')

    # Table B: every prompt conversion, mean foreground IoU (TC / AR)
    header = ['Prompter'] + [OUTPUTS[o] for o in OUTPUTS]
    body = []
    for meth in order:
        row = [METHODS[meth]]
        for out in OUTPUTS:
            if (meth, out) in m.index:
                row.append(f"{m.loc[(meth, out), 'TC IoU']:.2f} / {m.loc[(meth, out), 'AR IoU']:.2f}")
            else:
                row.append('--')
        body.append(row)
    write_table(f'prompt_conversion_{tag}', header, body,
                f'TC IoU / AR IoU for every way of turning the prompter output into SAM prompts ({ENCODER_NAMES[encoder]}).',
                f'tab:prompt_conversion_{tag}')

    # Table C: object-level detection of the own masks
    obj_cols = ['TC recall', 'TC precision', 'AR recall', 'AR precision']
    own = ev[ev['output'] == 'own']
    m2, s2, n2 = mean_std(own, ['method'], obj_cols + ['TC objects', 'AR objects'])
    body = [[METHODS[k]] + [fmt(m2.loc[k, c], s2.loc[k, c], n2.loc[k]) for c in obj_cols] +
            [f"{m2.loc[k, 'TC objects'] / 61:.1f}", f"{m2.loc[k, 'AR objects'] / 61:.1f}"] for k in order if k in m2.index]
    gt_obj = (f"{m2.loc['oracle_gt', 'TC objects'] / 61:.1f} TC and {m2.loc['oracle_gt', 'AR objects'] / 61:.1f} AR"
              if 'oracle_gt' in m2.index else 'the ground-truth number of')
    write_table(f'object_level_{tag}', ['Prompter'] + obj_cols + ['TC obj./img', 'AR obj./img'], body,
                'Object-level detection of the prompter masks: a ground-truth object is found if any predicted pixel '
                'overlaps it; a predicted object is correct if it overlaps any ground-truth object. Objects are '
                f'8-connected components of at least 20 pixels; with this counting the test set has {gt_obj} objects per image.',
                f'tab:object_level_{tag}')

    # Table D: error decomposition of the own masks
    dcols = ['as_is', 'no_false_objects', 'add_missed_objects', 'perfect_detection', 'perfect_shape_of_detected']
    body = []
    for cls in ('TC', 'AR'):
        d = dec[dec['class'] == cls]
        md, sd, nd = mean_std(d, ['method'], dcols)
        for k in order:
            if k in md.index and k != 'oracle_gt':
                body.append([METHODS[k], cls] + [fmt(md.loc[k, c], sd.loc[k, c], nd.loc[k]) for c in dcols])
        body.append('MIDRULE')
    write_table(f'error_decomposition_{tag}', ['Prompter', 'Class', 'As is', 'No false objects', 'Add missed objects',
                                               'Perfect detection', 'Perfect shape'], body[:-1],
                'IoU of the prompter mask after removing one error type: deleting predicted objects that touch no '
                'ground truth, adding the ground-truth objects that were missed, both, or giving every detected '
                'object its exact ground-truth shape.', f'tab:decomposition_{tag}', 'llccccc')

    # Table E: size and cost
    if not runs.empty:
        g = runs.groupby('method').agg(params=('trainable_params', 'first'), minutes=('train_time_min', 'mean'),
                                       epoch=('best_epoch', 'mean'))
        body = [[METHODS[k], f"{int(g.loc[k, 'params']):,}", f"{g.loc[k, 'minutes']:.1f}", f"{g.loc[k, 'epoch']:.0f}"]
                for k in order if k in g.index]
        write_table(f'cost_{tag}', ['Prompter', 'Trainable parameters', 'Training time (min)', 'Selected epoch'], body,
                    'Size and training cost (single A100, cached encoder features, 60 epochs; two-stage: 30 extra epochs '
                    'on top of the segmentation-loss model).', f'tab:cost_{tag}')


# ------------------------------------------------------------
# FIGURES
# ------------------------------------------------------------
def savefig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f'{name}.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_methods(encoder, ev):
    if ev.empty:
        return
    order = [k for k in METHODS if k in set(ev['method']) and k != 'oracle_gt']
    outs = ['own', 'sam_bbox', 'sam_point', 'sam_hybrid', 'sam_static']
    m, s, _ = mean_std(ev, ['method', 'output'], ['TC IoU', 'AR IoU'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    colors = ['#444444', '#e07b39', '#c9a227', '#3a7dc9', '#8e5ea2']
    for ax, cls in zip(axes, ('TC IoU', 'AR IoU')):
        x = np.arange(len(order))
        for j, out in enumerate(outs):
            vals = [m.loc[(k, out), cls] if (k, out) in m.index else np.nan for k in order]
            errs = [s.loc[(k, out), cls] if (k, out) in s.index else 0 for k in order]
            ax.bar(x + (j - 2) * 0.17, vals, 0.17, yerr=errs, label=OUTPUTS[out], color=colors[j], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[k] for k in order], rotation=35, ha='right', fontsize=8)
        ax.set_title(cls.replace(' IoU', ''))
        ax.set_ylabel('IoU')
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Prompter mask vs. SAM prompted by it ({ENCODER_NAMES[encoder]})')
    savefig(fig, f'methods_{encoder}')


def fig_curves(encoder, runs, logs):
    if runs.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for meth in [k for k in METHODS if k in set(runs['method'])]:
        names = runs[runs['method'] == meth]['name']
        for col, ax in (('val_own_Mean FG IoU', axes[0]), ('train_total', axes[1])):
            curves = [logs[n].dropna(subset=[col])[['epoch', col]] for n in names if col in logs[n]]
            if not curves:
                continue
            merged = pd.concat(curves).groupby('epoch')[col]
            mu, sd = merged.mean(), merged.std().fillna(0)
            line, = ax.plot(mu.index, mu.values, label=METHODS[meth])
            ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.2, color=line.get_color())
    axes[0].set_title('Validation mean FG IoU of the prompter mask')
    axes[1].set_title('Training loss')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    savefig(fig, f'training_curves_{encoder}')


def fig_decomposition(encoder, dec):
    if dec.empty:
        return
    order = [k for k in METHODS if k in set(dec['method']) and k != 'oracle_gt']
    cols = ['as_is', 'no_false_objects', 'add_missed_objects', 'perfect_detection', 'perfect_shape_of_detected']
    labels = ['as is', 'no false objects', 'add missed objects', 'perfect detection', 'perfect shape of detected']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, cls in zip(axes, ('TC', 'AR')):
        md = dec[dec['class'] == cls].groupby('method')[cols].mean()
        x = np.arange(len(order))
        for j, c in enumerate(cols):
            ax.bar(x + (j - 2) * 0.16, [md.loc[k, c] for k in order], 0.16, label=labels[j])
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[k] for k in order], rotation=35, ha='right', fontsize=8)
        ax.set_title(f'{cls}: IoU after fixing one error type')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('IoU')
    axes[0].legend(fontsize=8)
    savefig(fig, f'error_decomposition_{encoder}')


def fig_oracle(encoder):
    f = os.path.join(RESULTS, '01_oracle_prompts', f'oracle_sweep_{encoder}.csv')
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    for study in ('mask_format', 'degradation'):
        d = df[df['study'] == study]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d['TC IoU'], 0.4, label='TC', color=TC_COLOR)
        ax.bar(x + 0.2, d['AR IoU'], 0.4, label='AR', color=AR_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(d['prompt'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('IoU')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        ax.set_title({'mask_format': 'Ground-truth prompts: how the decoder reads dense mask prompts',
                      'degradation': 'Ground-truth prompts with controlled errors'}[study] + f' ({ENCODER_NAMES[encoder]})')
        savefig(fig, f'oracle_{study}_{encoder}')


def table413():
    """Corrected Table 4.13 in the thesis format (primary encoder), original-script values alongside."""
    f = os.path.join(RESULTS, '02_table_4_13_recheck', 'table_4_13_recheck.csv')
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    for enc, d in df.groupby('encoder'):
        tag = 'nosmooth' if 'NOSMOOTH' in enc else 'corrected'
        base = d[d['path'] == '-'].iloc[0]
        fixed = d[d['path'] == 'fixed'].set_index('prompt')
        orig = d[d['path'] == 'original'].set_index('prompt')
        fixed = fixed.sort_values('TC IoU', ascending=False)
        body = [['CG-Net alone (baseline)', '', '', '', f"{base['AR IoU']:.4f}", f"{base['TC IoU']:.4f}", '', ''], 'MIDRULE']
        for prompt, r in fixed.iterrows():
            m = re.match(r'(\S+) pos=(\d+) neg=(\d+) enlarge=(\S+)', prompt)
            t, p_, n_, e = m.groups()
            body.append([t, p_, n_, e, f"{r['AR IoU']:.4f}", f"{r['TC IoU']:.4f}",
                         f"{orig.loc[prompt, 'AR IoU']:.4f}", f"{orig.loc[prompt, 'TC IoU']:.4f}"])
        write_table(f'table_4_13_{tag}', ['Prompt type', 'Pos.', 'Neg.', 'Enlarge', 'IoU AR', 'IoU TC',
                                          'IoU AR (orig. script)', 'IoU TC (orig. script)'], body,
                    f'SAM with CG-Net as prompter, re-run on the 61 test images (Infused Token MLP 1.0, '
                    f'{"trained without label smoothing" if tag == "nosmooth" else "Phase-1 checkpoint of Table 4.12"}). '
                    "``orig. script'': image passed through "
                    'ClimateSAM.set\\_infer\\_img, which skips the learned input adapter (test\\_prompt\\_effect.py); '
                    'other columns: input adapter applied as in training. Ordered by TC IoU.',
                    f'tab:cgnet_prompter_{tag}', 'lccccccc')


def fig_table413():
    f = os.path.join(RESULTS, '02_table_4_13_recheck', 'table_4_13_recheck.csv')
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    for enc, d in df.groupby('encoder'):
        base = d[d['path'] == '-'].iloc[0]
        d = d[d['path'] != '-']
        prompts = list(dict.fromkeys(d['prompt']))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        for ax, cls in zip(axes, ('TC IoU', 'AR IoU')):
            x = np.arange(len(prompts))
            for j, path in enumerate(('original', 'fixed')):
                vals = [d[(d['prompt'] == p) & (d['path'] == path)][cls].values[0] for p in prompts]
                ax.bar(x + (j - 0.5) * 0.4, vals, 0.4, label={'original': 'original script (input adapter skipped)',
                                                             'fixed': 'fixed (input adapter applied)'}[path])
            ax.axhline(base[cls], color='k', ls='--', lw=1, label='CG-Net alone')
            ax.set_xticks(x)
            ax.set_xticklabels(prompts, rotation=60, ha='right', fontsize=7)
            ax.set_title(cls)
            ax.grid(axis='y', alpha=0.3)
        axes[0].legend(fontsize=8)
        savefig(fig, f'table_4_13_recheck_{os.path.basename(enc)}')


def draw_map(ax, gt, pred=None, boxes=None, title=''):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lons = np.linspace(0, 360, gt.shape[1], endpoint=False)
    lats = np.linspace(-90, 90, gt.shape[0])
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    for label, color in ((1, TC_COLOR), (2, AR_COLOR)):
        ax.contourf(lons, lats, (gt == label).astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.45,
                    transform=ccrs.PlateCarree())
    if pred is not None:
        for key, color in (('TC', '#d62728'), ('AR', '#ff7f0e')):
            if key in pred and pred[key].any():
                ax.contour(lons, lats, pred[key].astype(float), levels=[0.5], colors=[color], linewidths=0.8,
                           transform=ccrs.PlateCarree())
    for (cls, bx) in (boxes or []):
        for x1, y1, x2, y2 in bx:
            ax.add_patch(Rectangle((x1 / 1152 * 360, y1 / 768 * 180 - 90), (x2 - x1) / 1152 * 360, (y2 - y1) / 768 * 180,
                                   fill=False, lw=0.6, ec='#d62728' if cls == 'TC' else '#ff7f0e',
                                   transform=ccrs.PlateCarree()))
    ax.set_title(title, fontsize=8)


def fig_examples(encoder, methods=('oracle_gt', 'cgnet_finetuned', 'msf_seg_s0', 'msf_token_seg_s0', 'mpg_seg_s0',
                                   'mpg_twostage_s0')):
    import cartopy.crs as ccrs
    paths = {m: os.path.join(RESULTS, 'eval', encoder, f'{m}_examples.npz') for m in methods}
    paths = {m: p for m, p in paths.items() if os.path.exists(p)}
    if not paths:
        return
    loaded = {m: np.load(p) for m, p in paths.items()}
    images = sorted({int(k.split('__')[0]) for k in next(iter(loaded.values())).files})
    for img in images:
        fig, axes = plt.subplots(len(loaded), 3, figsize=(15, 2.6 * len(loaded)),
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        axes = np.atleast_2d(axes)
        for r, (meth, z) in enumerate(loaded.items()):
            get = lambda k: z[f'{img}__{k}'] if f'{img}__{k}' in z.files else None
            gt = get('gt')
            name = METHODS.get(method_key(meth), meth)
            draw_map(axes[r, 0], gt, {'TC': get('own_TC'), 'AR': get('own_AR')}, title=f'{name}: prompter mask')
            for c, kind in ((1, 'bbox'), (2, 'hybrid')):
                boxes = [(cls, get(f'boxes_{kind}_{cls}')) for cls in ('TC', 'AR') if get(f'boxes_{kind}_{cls}') is not None]
                draw_map(axes[r, c], gt, {'TC': get(f'sam_{kind}_TC'), 'AR': get(f'sam_{kind}_AR')}, boxes,
                         title=f'{name}: {OUTPUTS["sam_" + kind]}')
        fig.suptitle(f'Test image {img}: filled = ground truth (TC green, AR blue); lines = prediction (TC red, AR orange); '
                     'rectangles = box prompts', fontsize=10)
        savefig(fig, f'examples_{encoder}_img{img}')


def load_counts(encoder):
    """method -> output -> (n_images, 9) per-image [tp, fp, fn] x (TC, AR, BG), summed over seeds."""
    counts = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RESULTS, 'eval', encoder, '*.json'))):
        d = json.load(open(f))
        if 'per_image_counts' not in d:
            continue
        key = method_key(os.path.basename(f)[:-5])
        for out, arr in d['per_image_counts'].items():
            a = np.array(arr, dtype=np.float64)
            counts[key][out] = counts[key].get(out, 0) + a
    for f in sorted(glob.glob(os.path.join(RESULTS, 'runs', encoder, 'learned_prompt_*', 'test_per_image_counts.json'))):
        key = method_key(os.path.basename(os.path.dirname(f)))
        counts[key]['sam_static'] = counts[key].get('sam_static', 0) + np.array(json.load(open(f)), dtype=np.float64)
    return counts


def iou_from(c):
    """c: (..., 9) summed counts -> TC, AR IoU."""
    tc = c[..., 0] / np.maximum(c[..., 0] + c[..., 1] + c[..., 2], 1)
    ar = c[..., 3] / np.maximum(c[..., 3] + c[..., 4] + c[..., 5], 1)
    return tc, ar


def bootstrap(a, b, n=2000, seed=0):
    """Paired bootstrap over test images of (IoU of b) - (IoU of a); returns dict of mean / CI / P(>0)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    ta, aa = iou_from(a[idx].sum(1))
    tb, ab = iou_from(b[idx].sum(1))
    out = {}
    for name, d in (('TC', tb - ta), ('AR', ab - aa), ('FG', (tb + ab - ta - aa) / 2)):
        t0a, a0a = iou_from(a.sum(0))
        t0b, a0b = iou_from(b.sum(0))
        point = {'TC': t0b - t0a, 'AR': a0b - a0a, 'FG': (t0b + a0b - t0a - a0a) / 2}[name]
        out[name] = (point, np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean())
    return out


def bootstrap_table(encoder):
    counts = load_counts(encoder)
    if not counts:
        return
    comps = []
    for m in METHODS:
        if m == 'oracle_gt' or m not in counts:
            continue
        for out in ('sam_bbox', 'sam_hybrid', 'fused_hybrid'):
            if out in counts[m]:
                comps.append((f'{METHODS[m]}: {OUTPUTS[out]} vs. prompter mask', (m, 'own'), (m, out)))
    pairs = [('mpg_seg', 'cgnet_finetuned'), ('mpg_seg', 'msf_seg'), ('mpg_seg', 'logreg_last_seg'),
             ('msf_token_seg', 'msf_seg'), ('logreg_last_seg', 'logreg_l0_seg'), ('mpg_seg_smooth', 'mpg_seg'),
             ('msf_seg', 'cgnet_finetuned')]
    for b, a in pairs:
        if a in counts and b in counts:
            comps.append((f'{METHODS.get(b, b)} vs. {METHODS.get(a, a)} (prompter masks)', (a, 'own'), (b, 'own')))
    for b in ('mpg_twostage', 'mpg_sam_e2e'):
        if b in counts and 'mpg_seg' in counts:
            comps.append((f'{METHODS[b]}: SAM + hybrid vs. segmentation-loss generator mask', ('mpg_seg', 'own'), (b, 'sam_hybrid')))
    for b in ('learned_prompt_k4', 'learned_prompt_k16'):
        if b in counts and 'mpg_seg' in counts:
            comps.append((f'{METHODS[b]} vs. segmentation-loss generator mask', ('mpg_seg', 'own'), (b, 'sam_static')))
    body, rows = [], []
    for label, (ma, oa), (mb, ob) in comps:
        a, b = counts[ma][oa], counts[mb][ob]
        # IoU is a ratio of summed counts, so pooling a different number of seeds on each side does not bias it
        r = bootstrap(a, b)
        cells = [f'{r[k][0]:+.3f} [{r[k][1]:+.3f}, {r[k][2]:+.3f}]' for k in ('TC', 'AR', 'FG')]
        body.append([label] + cells + [f"{r['FG'][3]:.2f}"])
        rows.append({'comparison': label, **{f'{k} {n}': r[k][i] for k in ('TC', 'AR', 'FG')
                                               for i, n in enumerate(('delta', 'ci_low', 'ci_high', 'p_positive'))}})
    write_table(f'bootstrap_{encoder}', ['Comparison (B minus A)', '$\\Delta$ TC IoU [95\\% CI]', '$\\Delta$ AR IoU [95\\% CI]',
                                         '$\\Delta$ mean FG IoU [95\\% CI]', 'P($\\Delta$FG$>$0)'], body,
                f'Paired bootstrap over the 61 test images (2000 resamples; learned prompters: the three seeds pooled). '
                f'{ENCODER_NAMES[encoder]}.', f'tab:bootstrap_{encoder}', 'lcccc')


def decoder_table(encoder):
    runs = sorted(glob.glob(os.path.join(RESULTS, 'runs', encoder, 'decoder_adapt_*', 'summary.json')))
    if not runs:
        return
    labels = {'cgnet_finetuned': 'CG-Net (fine-tuned)', 'mpg_seg_s0': 'Mask-prompt generator'}
    body, fig, ax = [], *plt.subplots(figsize=(7, 4))
    for f in runs:
        d = json.load(open(f))
        oof = 'out-of-fold' if '_oof' in d['name'] else 'in-sample'
        head = f"{labels.get(d['prompter'], d['prompter'])}, {d['kind']}, {oof}"
        for k, v in d['test'].items():
            obj = (f"{v['TC recall']:.2f} / {v['TC precision']:.2f}", f"{v['AR recall']:.2f} / {v['AR precision']:.2f}") \
                if 'TC recall' in v else ('', '')
            body.append([head if k == 'prompter mask' else '', k, f"{v['TC IoU']:.3f}", f"{v['AR IoU']:.3f}",
                         f"{v['Mean FG IoU']:.3f}", *obj])
        body.append('MIDRULE')
        log = pd.read_csv(os.path.join(os.path.dirname(f), 'log.csv'))
        ax.plot(log['epoch'], log['val_Mean FG IoU'], marker='o', ms=2, label=head)
    write_table(f'decoder_adaptation_{encoder}', ['Prompter, prompts, training prompts', 'Output', 'TC IoU', 'AR IoU',
                                                  'Mean FG IoU', 'TC recall / prec.', 'AR recall / prec.'], body[:-1],
                'Adapting the HQ decoder parts to generated prompts (test set). Phase-1 decoder = the frozen decoder of '
                'all other experiments; adapted = best validation epoch of 30.', f'tab:decoder_adaptation_{encoder}',
                'llccccc')
    ax.set_xlabel('epoch (0 = Phase-1 decoder)')
    ax.set_ylabel('validation mean FG IoU (SAM output)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    savefig(fig, f'decoder_adaptation_{encoder}')


def fig_sam_minus_prompter(encoder):
    """SAM output minus the prompter's own mask, with paired-bootstrap 95% intervals (the central Section 4.3 claim)."""
    counts = load_counts(encoder)
    order = [k for k in METHODS if k in counts and k != 'oracle_gt' and 'own' in counts[k]]
    if not order:
        return
    outs = [('sam_bbox', '#e07b39'), ('sam_point', '#c9a227'), ('sam_hybrid', '#3a7dc9')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(order) + 1.5), sharey=True)
    for ax, metric in zip(axes, ('TC', 'AR', 'FG')):
        for j, (out, color) in enumerate(outs):
            for i, k in enumerate(order):
                if out not in counts[k]:
                    continue
                r = bootstrap(counts[k]['own'], counts[k][out])[metric]
                y = i + (j - 1) * 0.25
                ax.errorbar(r[0], y, xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt='o', color=color, ms=4, capsize=2,
                            label=OUTPUTS[out] if i == 0 else None)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title({'TC': 'TC IoU', 'AR': 'AR IoU', 'FG': 'mean FG IoU'}[metric] + ': SAM minus prompter mask')
        ax.grid(axis='x', alpha=0.3)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([METHODS[k] for k in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f'Does SAM improve on its prompter? Paired bootstrap 95% intervals, 61 test images ({ENCODER_NAMES[encoder]})')
    savefig(fig, f'sam_minus_prompter_{encoder}')


def exploratory():
    """Curves + summary of the first exploratory runs (encoder retrain_infused_05, BCE loss, selection on test)."""
    d = os.path.join(RESULTS, '07_exploratory_runs')
    runs = {'run_v1': 'end-to-end via SAM (from scratch)', 'run_ml4': 'end-to-end, 4 ViT blocks',
            'run_segonly': 'segmentation loss only', 'run_finetune': 'two-stage (seg. loss -> via SAM)'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    rows = []
    for key, label in runs.items():
        f = os.path.join(d, 'logs', f'{key}.log')
        if not os.path.exists(f):
            continue
        epoch, vals = 0, []
        for line in open(f):
            m = re.match(r'Epoch (\d+)', line)
            if m:
                epoch = int(m.group(1))
            m = re.search(r'SAM IoU TC: ([\d.]+)%, AR: ([\d.]+)% \| generator IoU TC: ([\d.]+)%, AR: ([\d.]+)%', line)
            if m:
                vals.append((epoch, *[float(x) / 100 for x in m.groups()]))
        if not vals:
            continue
        v = np.array(vals)
        for ax, (i, j), title in ((axes[0], (1, 2), 'SAM output'), (axes[1], (3, 4), 'generator mask')):
            line, = ax.plot(v[:, 0], (v[:, i] + v[:, j]) / 2, marker='o', ms=3, label=label)
            ax.set_title(f'Mean FG IoU of the {title} (test set)')
        best = v[np.argmax(v[:, 1] + v[:, 2])]
        rows.append([label, f'{int(best[0])}', *[f'{x:.3f}' for x in best[1:]]])
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    savefig(fig, 'exploratory_runs')
    write_table('exploratory_runs', ['Run', 'Best epoch', 'SAM TC', 'SAM AR', 'Generator TC', 'Generator AR'], rows,
                'First exploratory runs of the mask-prompt generator (encoder retrain\\_infused\\_05, BCE + Tversky loss, '
                'selected on the test set -- optimistic, superseded by Section 04).', 'tab:exploratory')


def speed_table():
    f = os.path.join(RESULTS, '04_sam_feature_prompters', 'speed.csv')
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    body = [[r['prompter'], f"{int(r['params']):,}", f"{r['GFLOPs']:.2f}", f"{r['latency_ms']:.2f}"] for _, r in d.iterrows()]
    write_table('speed', ['Model', 'Parameters', 'GFLOPs / image', 'Latency (ms, A100, batch 1)'], body,
                'Cost of the prompters on top of the frozen ClimateSAM encoder (forward pass, one image; FLOPs counted '
                'with torch.utils.flop\\_counter, latency = median of 20 runs). The encoder and one decoder call are '
                'listed for reference.', 'tab:speed', 'lccc')


def render_readme():
    """results/README.template.md -> results/README.md, with {{table:NAME}} replaced by tables/NAME.md."""
    tpl = os.path.join(RESULTS, 'README.template.md')
    if not os.path.exists(tpl):
        return
    text = open(tpl).read()
    for name in re.findall(r'\{\{table:([\w.-]+)\}\}', text):
        f = os.path.join(TABLES, f'{name}.md')
        text = text.replace(f'{{{{table:{name}}}}}', open(f).read() if os.path.exists(f) else f'*(table {name} not generated yet)*')
    with open(os.path.join(RESULTS, 'README.md'), 'w') as f:
        f.write('<!-- generated from README.template.md by prompter_bench/make_report.py -->\n' + text)


def main():
    exploratory()
    for encoder in ENCODER_NAMES:
        ev, dec = load_eval(encoder)
        runs, logs = load_runs(encoder)
        if not ev.empty:
            ev.to_csv(os.path.join(RESULTS, 'eval', encoder, 'all_results.csv'), index=False)
            dec.to_csv(os.path.join(RESULTS, 'eval', encoder, 'all_error_decomposition.csv'), index=False)
        if not runs.empty:
            runs.drop(columns=['args']).to_csv(os.path.join(RESULTS, 'runs', encoder, 'all_runs.csv'), index=False)
        main_tables(encoder, ev, dec, runs)
        fig_methods(encoder, ev)
        fig_curves(encoder, runs, logs)
        fig_decomposition(encoder, dec)
        fig_oracle(encoder)
        fig_examples(encoder)
        bootstrap_table(encoder)
        fig_sam_minus_prompter(encoder)
        decoder_table(encoder)
    fig_table413()
    table413()
    speed_table()
    render_readme()


if __name__ == '__main__':
    main()
