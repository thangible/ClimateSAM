"""
All figures of the Section 4.3 report in one visual grammar (style.py). Reads only the raw outputs in results/.
Every multi-panel figure is also written panel by panel to results/figures/panels/.
"""
import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import style as S
from style import figure, ygrid
from common import RESULTS
import make_report as mr

FIG = os.path.join(RESULTS, 'figures')
ENC = 'infused_mlp1'
SHORT = {
    'cgnet_official': 'CG-Net (official)', 'cgnet_finetuned': 'CG-Net (fine-tuned)',
    'cgnet_train_seg': 'CG-Net (re-trained)', 'cgnet_scratch_seg': 'CG-Net (from scratch)',
    'msf_token_cg_seg': 'MSF + token gate, CG blocks', 'msf_sp_token_seg': 'MSF + token gate, shared',
    'mpg_fields_seg': 'MPG + raw fields', 'det_head': 'Box head on SAM features',
    'logreg_l0_seg': 'LogReg, block 1', 'logreg_last_seg': 'LogReg, block 12',
    'msf_seg': 'MSF', 'msf_token_seg': 'MSF + token gate',
    'mpg_seg': 'MPG', 'mpg_seg_smooth': 'MPG + label smooth.', 'mpg_sam_e2e': 'MPG end-to-end',
    'mpg_twostage': 'MPG two-stage', 'learned_prompt_k4': 'Static prompts (4)', 'learned_prompt_k16': 'Static prompts (16)',
}
PROMPTERS = ['cgnet_official', 'cgnet_finetuned', 'cgnet_train_seg', 'cgnet_scratch_seg', 'logreg_l0_seg',
             'logreg_last_seg', 'msf_seg', 'msf_token_seg', 'msf_token_cg_seg', 'msf_sp_token_seg',
             'mpg_seg', 'mpg_seg_smooth', 'mpg_sam_e2e', 'mpg_twostage', 'mpg_fields_seg']
OUT_LABEL = {'own': 'Prompter mask (no SAM)', 'sam_bbox': 'SAM + box', 'sam_hybrid': 'SAM + hybrid',
             'sam_point': 'SAM + points', 'sam_static': 'SAM + learned static prompts', 'adapted': 'SAM, adapted decoder'}


def is_mpg(k):
    return k == 'mpg_seg'


def bar_handles(keys):
    return [(Patch(color=S.OUTPUT[k]), OUT_LABEL[k]) for k in keys]


def dot_handles(keys):
    return [(Line2D([], [], marker='o', ls='', color=S.OUTPUT[k]), OUT_LABEL[k]) for k in keys]


def class_handles():
    return [(Patch(color=S.TC), 'TC'), (Patch(color=S.AR), 'AR')]


def label_ends(ax, items, x, min_gap):
    """
    Direct labels at the right end of lines; items = [(y, text, color)] or [(y, text, color, x_end)], nudged apart
    vertically so they never overlap.
    """
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for item, y in zip(items, ys):
        y0, text, color = item[:3]
        xx = item[3] if len(item) > 3 else x
        ax.annotate(text, (xx, y0), xytext=(xx, y), textcoords='data', fontsize=6.5, color=color, va='center',
                    annotation_clip=False)


# ------------------------------------------------------------
# 1  ORACLE PROMPTS
# ------------------------------------------------------------
def fig_oracle():
    ev, _ = mr.load_eval(ENC)
    o = ev[ev['method'] == 'oracle_gt'].set_index('output')
    kinds = [('sam_bbox', 'box'), ('sam_bbox_e10', 'box +10%'), ('sam_point', 'points'), ('sam_bbox+point', 'box + points'),
             ('sam_mask', 'mask logits'), ('sam_bbox+mask', 'box + mask'), ('sam_hybrid', 'hybrid')]

    def draw(ax):
        x = np.arange(len(kinds))
        for j, cls in enumerate(('TC', 'AR')):
            ax.bar(x + (j - 0.5) * 0.38, [o.loc[k, f'{cls} IoU'] for k, _ in kinds], 0.36, color=S.CLASS[cls],
                   edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in kinds], rotation=30, ha='right')
        ax.set_ylabel('IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return class_handles()
    figure(FIG, 'oracle_prompt_types', [('all', 'SAM prompted with ground-truth prompts', draw)], size=(6.6, 2.6))

    # mask-prompt format, both checkpoints
    a = pd.read_csv(os.path.join(RESULTS, '01_oracle_prompts', f'oracle_sweep_{ENC}.csv'))
    b = pd.read_csv(os.path.join(RESULTS, '01_oracle_prompts', 'oracle_sweep_infused_mlp05.csv'))
    fa, fb = a[a['study'] == 'mask_format'].set_index('prompt'), b[b['study'] == 'mask_format'].set_index('prompt')
    rows = list(fa.index)

    def fmt_panel(cls):
        def draw(ax):
            y = np.arange(len(rows))
            ax.scatter(fa.loc[rows, f'{cls} IoU'], y, color=S.CLASS[cls], s=22, zorder=3)
            ax.scatter(fb.loc[rows, f'{cls} IoU'], y, facecolor='white', edgecolor=S.CLASS[cls], s=22, linewidth=1.2, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(rows)
            ax.invert_yaxis()
            ax.set_xlim(-0.02, 1)
            ax.set_xlabel(f'{cls} IoU (test)')
            ygrid(ax, 'x')
            return [(Line2D([], [], marker='o', ls='', color=S.INK2), 'Infused Token MLP 1.0 (primary)'),
                    (Line2D([], [], marker='o', ls='', mfc='white', color=S.INK2), 'Infused Token MLP 0.5')]
        return draw
    figure(FIG, 'oracle_mask_format', [('tc', 'TC', fmt_panel('TC')), ('ar', 'AR', fmt_panel('AR'))],
           size=(3.3, 3.4), sharey=True)

    # controlled prompt errors
    d = a[a['study'] == 'degradation'].set_index('prompt')

    def scale(ax):
        rs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
        for cls in ('TC', 'AR'):
            ax.plot(rs, [d.loc[f'box enlarge {r:+.1f}', f'{cls} IoU'] for r in rs], marker='o', color=S.CLASS[cls])
        ax.axvline(0, color=S.AXIS, lw=0.8)
        ax.set_xlabel('box enlarged by (fraction of its size)')
        ax.set_ylabel('IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return class_handles()

    def morph(ax):
        ks = [-9, -5, -3, 0, 3, 5, 9]
        for cls in ('TC', 'AR'):
            ax.plot(ks, [d.loc[f'mask morph {k:+d}px', f'{cls} IoU'] for k in ks], marker='o', color=S.CLASS[cls])
        ax.axvline(0, color=S.AXIS, lw=0.8)
        ax.set_xticks(ks)
        ax.set_xlabel('mask prompt eroded (<0) / dilated (>0), px')
        ax.set_ylabel('IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return class_handles()

    def objects_(ax):
        cats = [('box enlarge +0.0', 'none'), ('box drop 25% objects', '25% not\nprompted'),
                ('box drop 50% objects', '50% not\nprompted'), ('box + 1 false boxes', '+1 false\nbox'),
                ('box + 3 false boxes', '+3 false\nboxes')]
        x = np.arange(len(cats))
        for j, cls in enumerate(('TC', 'AR')):
            ax.bar(x + (j - 0.5) * 0.38, [d.loc[c, f'{cls} IoU'] for c, _ in cats], 0.36, color=S.CLASS[cls],
                   edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in cats])
        ax.set_ylabel('IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return class_handles()
    figure(FIG, 'oracle_prompt_errors', [('box_size', 'Box size', scale), ('mask_morphology', 'Mask prompt thickness', morph),
                                         ('objects', 'Missed / false objects (boxes)', objects_)], size=(3.3, 2.5))


# ------------------------------------------------------------
# 2  TABLE 4.13 RE-CHECK
# ------------------------------------------------------------
def fig_table413():
    df = pd.read_csv(os.path.join(RESULTS, '02_table_4_13_recheck', 'table_4_13_recheck.csv'))
    d = df[df['encoder'] == 'infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED']
    base = d[d['path'] == '-'].iloc[0]
    fixed = d[d['path'] == 'fixed'].set_index('prompt').sort_values('TC IoU', ascending=False)
    orig = d[d['path'] == 'original'].set_index('prompt')
    rows = list(fixed.index)
    labels = [re.sub(r' pos=(\d+) neg=(\d+) enlarge=(\S+)',
                     lambda m: (f' ({m.group(1)}/{m.group(2)})' if m.group(1) != '0' else '') +
                               (f', {float(m.group(3)):+.1f}' if float(m.group(3)) else ''), r) for r in rows]

    def panel(cls):
        def draw(ax):
            y = np.arange(len(rows))
            ax.axvline(base[f'{cls} IoU'], color=S.INK2, lw=0.9)
            ax.text(base[f'{cls} IoU'] - 0.002, -1.1, 'CG-Net alone', fontsize=6.5, color=S.INK2, ha='right')
            ax.scatter(fixed.loc[rows, f'{cls} IoU'], y, color=S.CLASS[cls], s=18, zorder=3)
            ax.scatter(orig.loc[rows, f'{cls} IoU'], y, facecolor='white', edgecolor=S.CLASS[cls], s=18, lw=1.1, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=6.5)
            ax.set_ylim(len(rows) - 0.5, -1.6)
            ax.set_xlabel(f'{cls} IoU (test)')
            ygrid(ax, 'x')
            return [(Line2D([], [], marker='o', ls='', color=S.INK2), 'input adapter applied (fixed)'),
                    (Line2D([], [], marker='o', ls='', mfc='white', color=S.INK2), 'original script (adapter skipped)')]
        return draw
    figure(FIG, 'table_4_13_recheck', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR'))], size=(3.3, 4.2), sharey=True)


# ------------------------------------------------------------
# 3  PROMPTERS: OWN MASK vs SAM
# ------------------------------------------------------------
def fig_prompters():
    ev, _ = mr.load_eval(ENC)
    m, s, _ = mr.mean_std(ev, ['method', 'output'], ['TC IoU', 'AR IoU', 'Mean FG IoU'])
    methods = [k for k in PROMPTERS + ['learned_prompt_k4', 'learned_prompt_k16'] if k in set(ev['method'])]
    outs = ['own', 'sam_bbox', 'sam_hybrid', 'sam_point', 'sam_static']

    def panel(metric):
        def draw(ax):
            x = np.arange(len(methods))
            w = 0.16
            for j, out in enumerate(outs):
                vals = [m.loc[(k, out), metric] if (k, out) in m.index else np.nan for k in methods]
                errs = [s.loc[(k, out), metric] if (k, out) in s.index else 0 for k in methods]
                if out == 'sam_static':  # static prompts have only this output: centre it on its slot
                    pos = x.astype(float)
                else:
                    pos = x + (j - 1.5) * w
                ax.bar(pos, vals, w, yerr=errs, color=S.OUTPUT[out], edgecolor='white', linewidth=0.6,
                       error_kw=dict(ecolor=S.INK2, elinewidth=0.7))
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[k] for k in methods], rotation=30, ha='right')
            for t, k in zip(ax.get_xticklabels(), methods):
                if is_mpg(k):
                    t.set_color(S.EMPH)
                    t.set_fontweight('bold')
            ax.set_ylabel(metric.replace(' IoU', ' IoU (test)'))
            ax.set_ylim(0, 0.45)
            ygrid(ax)
            return bar_handles(outs)
        return draw
    figure(FIG, 'prompters_mask_vs_sam', [('tc', 'TC', panel('TC IoU')), ('ar', 'AR', panel('AR IoU')),
                                          ('fg', 'Mean of TC and AR', panel('Mean FG IoU'))],
           ncols=1, size=(6.6, 2.7), legend_ncol=5, sharex=True)


def fig_sam_minus_prompter():
    counts = mr.load_counts(ENC)
    methods = [k for k in PROMPTERS if k in counts]
    outs = ['sam_bbox', 'sam_hybrid', 'sam_point']

    def panel(metric):
        def draw(ax):
            for j, out in enumerate(outs):
                for i, k in enumerate(methods):
                    if out not in counts[k]:
                        continue
                    r = mr.bootstrap(counts[k]['own'], counts[k][out])[metric]
                    ax.errorbar(r[0], i + (j - 1) * 0.24, xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt='o', ms=3.5,
                                color=S.OUTPUT[out], elinewidth=1, capsize=1.5)
            ax.axvline(0, color=S.INK2, lw=0.9)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([SHORT[k] for k in methods])
            for t, k in zip(ax.get_yticklabels(), methods):
                if is_mpg(k):
                    t.set_color(S.EMPH)
                    t.set_fontweight('bold')
            ax.set_ylim(len(methods) - 0.5, -0.5)
            ax.set_xlabel(f'Δ {metric} IoU: SAM − prompter mask')
            ygrid(ax, 'x')
            return dot_handles(outs)
        return draw
    figure(FIG, 'sam_minus_prompter', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR')), ('fg', 'Mean of TC and AR', panel('FG'))],
           size=(3.3, 0.9 + 0.26 * len(methods)), sharey=True)


# ------------------------------------------------------------
# 4  ERROR ANALYSIS
# ------------------------------------------------------------
def fig_objects():
    ev, dec = mr.load_eval(ENC)
    main = ['cgnet_official', 'cgnet_finetuned', 'logreg_l0_seg', 'logreg_last_seg', 'msf_seg', 'msf_token_seg', 'mpg_seg']
    own = ev[(ev['output'] == 'own') & ev['method'].isin(main)]
    g = own.groupby('method')[['TC recall', 'TC precision', 'AR recall', 'AR precision']]
    m, s = g.mean(), g.std().fillna(0)
    off = {  # label offsets in points: (dx, dy, horizontal alignment)
        'TC': {'cgnet_finetuned': (6, -3, 'left'), 'cgnet_official': (-6, -9, 'right'), 'logreg_l0_seg': (6, 0, 'left'),
               'logreg_last_seg': (6, 4, 'left'), 'msf_seg': (-7, 0, 'right'), 'msf_token_seg': (6, -2, 'left'),
               'mpg_seg': (-7, 5, 'right')},
        'AR': {'cgnet_finetuned': (6, -6, 'left'), 'cgnet_official': (6, 5, 'left'), 'logreg_l0_seg': (-7, 0, 'right'),
               'logreg_last_seg': (-7, 0, 'right'), 'msf_seg': (-7, -5, 'right'), 'msf_token_seg': (-7, 0, 'right'),
               'mpg_seg': (-7, 5, 'right')},
    }

    def panel(cls):
        def draw(ax):
            for k in m.index:
                c = S.EMPH if is_mpg(k) else S.OTHER
                ax.errorbar(m.loc[k, f'{cls} recall'], m.loc[k, f'{cls} precision'], xerr=s.loc[k, f'{cls} recall'],
                            yerr=s.loc[k, f'{cls} precision'], fmt='o', color=c, ms=4.5, elinewidth=0.8, capsize=0)
                dx, dy, ha = off[cls][k]
                ax.annotate(SHORT[k], (m.loc[k, f'{cls} recall'], m.loc[k, f'{cls} precision']), xytext=(dx, dy),
                            textcoords='offset points', fontsize=6.5, ha=ha, va='center',
                            color=S.EMPH if is_mpg(k) else S.INK2)
            ax.set_xlabel(f'{cls} object recall')
            ax.set_ylabel(f'{cls} object precision')
            ax.set_xlim((0.55, 0.85) if cls == 'TC' else (0.85, 1.0))
            ax.set_ylim(0.2, 0.65)
            ygrid(ax, 'both')
            return None
        return draw
    figure(FIG, 'object_level', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR'))], size=(3.3, 2.8))

    d = dec[dec['method'].isin(PROMPTERS)]
    variants = [('as_is', 'as predicted'), ('perfect_detection', 'perfect detection'),
                ('perfect_shape_of_detected', 'perfect shape of detected objects')]
    methods = [k for k in PROMPTERS if k in set(d['method'])]

    def dpanel(cls):
        def draw(ax):
            md = d[d['class'] == cls].groupby('method')[[v for v, _ in variants]].mean()
            y = np.arange(len(methods))
            for j, (v, _) in enumerate(variants):
                ax.barh(y + (j - 1) * 0.27, [md.loc[k, v] for k in methods], 0.25, color=S.RAMP3[cls][j],
                        edgecolor='white', linewidth=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([SHORT[k] for k in methods])
            for t, k in zip(ax.get_yticklabels(), methods):
                if is_mpg(k):
                    t.set_color(S.EMPH)
                    t.set_fontweight('bold')
            ax.set_ylim(len(methods) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel(f'{cls} IoU of the prompter mask (test)')
            ygrid(ax, 'x')
            neutral = ['#c8c7c1', '#8a8881', '#3d3c39']  # legend in grey: the hue follows the class of the panel
            return [(Patch(color=neutral[j]), l) for j, (_, l) in enumerate(variants)]
        return draw
    figure(FIG, 'error_decomposition', [('tc', 'TC', dpanel('TC')), ('ar', 'AR', dpanel('AR'))], size=(3.3, 1.0 + 0.3 * len(methods)),
           sharey=True, legend_ncol=3)


# ------------------------------------------------------------
# 5  TRAINING, EFFICIENCY, ROBUSTNESS
# ------------------------------------------------------------
def fig_training():
    runs, logs = mr.load_runs(ENC)
    methods = ['logreg_l0_seg', 'logreg_last_seg', 'msf_seg', 'msf_token_seg', 'mpg_seg']

    def curve(col, log_y=False):
        def draw(ax):
            ends = []
            for k in methods:
                names = runs[runs['method'] == k]['name']
                cs = [logs[n].dropna(subset=[col])[['epoch', col]] for n in names]
                g = pd.concat(cs).groupby('epoch')[col]
                mu, sd = g.mean(), g.std().fillna(0)
                c = S.EMPH if is_mpg(k) else S.OTHER
                ax.plot(mu.index, mu.values, color=c, lw=1.8 if is_mpg(k) else 1.2)
                ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
                ends.append((mu.values[-1], SHORT[k], S.EMPH if is_mpg(k) else S.INK2))
            ax.set_xlabel('epoch')
            if log_y:
                ax.set_yscale('log')
            ygrid(ax)
            x_end = max(mu.index) + 1.5
            lo, hi = ax.get_ylim()
            label_ends(ax, ends, x_end, (hi - lo) * 0.055 if not log_y else 0)
            ax.set_xlim(0, max(mu.index) + 1)
            return None
        return draw
    figure(FIG, 'training_curves', [('val', 'Validation mean FG IoU (prompter mask)', curve('val_own_Mean FG IoU')),
                                    ('loss', 'Training loss (Tversky + Focal)', curve('train_total'))], size=(3.6, 2.6))


def fig_efficiency():
    sp = pd.read_csv(os.path.join(RESULTS, '04_sam_feature_prompters', 'speed.csv')).set_index('prompter')
    ev, _ = mr.load_eval(ENC)
    own = ev[ev['output'] == 'own'].groupby('method')['Mean FG IoU'].mean()
    pts = [('cgnet_finetuned', 'CG-Net'), ('logreg_last_seg', 'Logistic regression'), ('msf_seg', 'Multi-scale fusion'),
           ('msf_token_seg', 'Multi-scale fusion + token gate'), ('mpg_seg', 'Mask-prompt generator')]

    def draw(ax):
        offsets = {'cgnet_finetuned': (-6, -10, 'right'), 'logreg_last_seg': (6, 0, 'left'), 'msf_seg': (-6, 8, 'right'),
                   'msf_token_seg': (-6, -8, 'right'), 'mpg_seg': (6, 0, 'left')}
        for k, name in pts:
            x, y = sp.loc[name, 'GFLOPs'], own[k]
            c = S.EMPH if is_mpg(k) else S.OTHER
            ax.scatter(x, y, color=c, s=30, zorder=3)
            p_ = sp.loc[name, 'params']
            params = f'{p_ / 1e6:.2f} M' if p_ >= 1e5 else f'{p_ / 1e3:.1f} k'
            lat = sp.loc[name, 'latency_ms']
            dx, dy, ha = offsets[k]
            ax.annotate(f'{SHORT[k]}\n{params} params, {lat:.2f} ms' if lat < 1 else f'{SHORT[k]}\n{params} params, {lat:.1f} ms',
                        (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6, ha=ha, va='center',
                        color=S.EMPH if is_mpg(k) else S.INK2)
        ax.set_xscale('log')
        ax.set_xlim(5e-3, 5e3)
        ax.set_ylim(0.32, 0.39)
        ax.set_xlabel('GFLOPs per image (log scale; frozen encoder: 973)')
        ax.set_ylabel('Mean FG IoU of the prompter mask (test)')
        ygrid(ax, 'both')
        return None
    figure(FIG, 'efficiency', [('all', 'Accuracy vs. cost of the prompter', draw)], size=(4.6, 3.0))


def fig_second_encoder():
    a = pd.read_csv(os.path.join(RESULTS, 'eval', ENC, 'all_results.csv'))
    b = pd.read_csv(os.path.join(RESULTS, 'eval', 'infused_mlp05', 'all_results.csv'))
    ga = a.groupby(['method', 'output'])['Mean FG IoU'].mean()
    gb = b.groupby(['method', 'output'])['Mean FG IoU'].mean()
    outs = ['own', 'sam_bbox', 'sam_hybrid', 'sam_point']

    def draw(ax):
        for out in outs:
            keys = [k for k in PROMPTERS if (k, out) in ga.index and (k, out) in gb.index]
            ax.scatter([ga[(k, out)] for k in keys], [gb[(k, out)] for k in keys], color=S.OUTPUT[out], s=18, zorder=3,
                       edgecolor='white', linewidth=0.5)
        ax.plot([0.2, 0.4], [0.2, 0.4], color=S.AXIS, lw=0.9, zorder=1)
        ax.set_xlim(0.2, 0.4)
        ax.set_ylim(0.2, 0.4)
        ax.set_aspect('equal')
        ax.set_xlabel('Mean FG IoU, Infused Token MLP 1.0')
        ax.set_ylabel('Mean FG IoU, Infused Token MLP 0.5')
        ygrid(ax, 'both')
        return dot_handles(outs)
    figure(FIG, 'second_checkpoint', [('all', 'Same prompter and output on both frozen checkpoints', draw)],
           size=(3.4, 3.4), legend_ncol=2)


# ------------------------------------------------------------
# 6  DECODER ADAPTATION, YOLO, EXPLORATORY
# ------------------------------------------------------------
def fig_decoder():
    runs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, 'runs', ENC, 'decoder_adapt_*', 'summary.json'))):
        d = json.load(open(f))
        who = 'CG-Net' if d['prompter'].startswith('cgnet') else 'MPG'
        oof = 'out-of-fold' if '_oof' in d['name'] else 'in-sample'
        runs.append((f'{who}, {d["kind"]}, {oof}', d, os.path.join(os.path.dirname(f), 'log.csv')))

    def bars(ax):
        y = np.arange(len(runs))
        for i, (label, d, _) in enumerate(runs):
            t = d['test']
            v = [t['prompter mask']['Mean FG IoU'], t['SAM, Phase-1 decoder']['Mean FG IoU'], t['SAM, adapted decoder']['Mean FG IoU']]
            phase1 = S.OUTPUT['sam_bbox'] if d['kind'] == 'bbox' else S.OUTPUT['sam_hybrid']
            ax.plot([min(v), max(v)], [i, i], color=S.GRID, lw=2, zorder=1)
            for val, c in zip(v, [S.OUTPUT['own'], phase1, S.OUTPUT['adapted']]):
                ax.scatter(val, i, color=c, s=28, zorder=3, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in runs])
        ax.set_ylim(len(runs) - 0.5, -0.5)
        ax.set_xlabel('Mean FG IoU (test)')
        ygrid(ax, 'x')
        return [(Line2D([], [], marker='o', ls='', color=S.OUTPUT['own']), 'Prompter mask'),
                (Line2D([], [], marker='o', ls='', color=S.OUTPUT['sam_bbox']), 'SAM, Phase-1 decoder (box)'),
                (Line2D([], [], marker='o', ls='', color=S.OUTPUT['sam_hybrid']), 'SAM, Phase-1 decoder (hybrid)'),
                (Line2D([], [], marker='o', ls='', color=S.OUTPUT['adapted']), 'SAM, adapted decoder')]

    def curves(ax):
        ends = []
        for label, d, log in runs:
            l = pd.read_csv(log)
            ax.plot(l['epoch'], l['val_Mean FG IoU'], color=S.OUTPUT['adapted'], lw=1.1, alpha=0.9)
            ends.append((l['val_Mean FG IoU'].values[-1], label, S.INK2))
        ax.set_xlabel('epoch (0 = Phase-1 decoder)')
        ax.set_ylabel('Validation mean FG IoU (SAM)')
        ygrid(ax)
        lo, hi = ax.get_ylim()
        label_ends(ax, ends, 31, (hi - lo) * 0.06)
        ax.set_xlim(0, 30)
        return None
    figure(FIG, 'decoder_adaptation', [('test', 'Test set', bars), ('val', 'Validation during fine-tuning', curves)],
           size=(3.6, 2.6), legend_ncol=2)


def fig_yolo():
    d = pd.read_csv(os.path.join(RESULTS, '03_cgnet_yolo_boxes', f'cgnet_yolo_{ENC}.csv'))
    d = d[d['output'] == 'sam_bbox']
    ev, _ = mr.load_eval(ENC)
    cg = ev[ev['method'] == 'cgnet_finetuned'].set_index('output')
    names = {'exp/cgnet_bbox_weight.pth': 'checkpoint A', 'exp/best_weights/cgnet_bbox_weight.pth': 'checkpoint B',
             'exp/best_weights/best_exp_cgnet_bbox_weight.pth': 'checkpoint C'}

    def panel(cls):
        def draw(ax):
            ends = []
            for w, g in d.groupby('weights'):
                ax.plot(g['conf_threshold'], g[f'{cls} IoU'], marker='o', color=S.OUTPUT['sam_bbox'], lw=1.1, alpha=0.85)
                ends.append((g[f'{cls} IoU'].values[-1], names[w], S.INK2))
            label_ends(ax, ends, 0.72, 0.014)
            ax.axhline(cg.loc['own', f'{cls} IoU'], color=S.OUTPUT['own'], lw=1.1)
            ax.text(0.3, cg.loc['own', f'{cls} IoU'] + 0.006, 'CG-Net segmentation mask', fontsize=6, color=S.INK2)
            ax.set_xlabel('objectness threshold')
            ax.set_ylabel(f'{cls} IoU (test)')
            ax.set_xlim(0.27, 0.85)
            ax.set_ylim(0.1, 0.42)
            ygrid(ax)
            return [(Line2D([], [], color=S.OUTPUT['sam_bbox'], marker='o'), 'SAM + YOLO boxes'),
                    (Line2D([], [], color=S.OUTPUT['own']), 'CG-Net mask (no SAM)')]
        return draw
    figure(FIG, 'cgnet_yolo', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR'))], size=(3.3, 2.5))


def fig_exploratory():
    runs = {'run_segonly': ('segmentation loss only', True), 'run_finetune': ('two-stage', False),
            'run_v1': ('end-to-end via SAM', False), 'run_ml4': ('end-to-end, 4 ViT blocks', False)}
    data = {}
    for key, (label, _) in runs.items():
        epoch, vals = 0, []
        for line in open(os.path.join(RESULTS, '07_exploratory_runs', 'logs', f'{key}.log')):
            mm = re.match(r'Epoch (\d+)', line)
            if mm:
                epoch = int(mm.group(1))
            mm = re.search(r'SAM IoU TC: ([\d.]+)%, AR: ([\d.]+)% \| generator IoU TC: ([\d.]+)%, AR: ([\d.]+)%', line)
            if mm:
                vals.append((epoch, *[float(x) / 100 for x in mm.groups()]))
        data[key] = np.array(vals)

    # label offsets (points) so no label sits on another run's line
    dys = {1: {'run_segonly': 0, 'run_finetune': 8, 'run_v1': 0, 'run_ml4': 8},
           3: {'run_segonly': 0, 'run_finetune': -2, 'run_v1': 6, 'run_ml4': -9}}

    def panel(i, j):
        def draw(ax):
            for key, (label, emph) in runs.items():
                v = data[key]
                c = S.EMPH if emph else S.OTHER
                y = (v[:, i] + v[:, j]) / 2
                ax.plot(v[:, 0], y, marker='o', ms=2.5, color=c, lw=1.4 if emph else 1.0)
                ax.annotate(label, (v[-1, 0], y[-1]), xytext=(4, dys[i][key]), textcoords='offset points', fontsize=6.5,
                            va='center', color=S.EMPH if emph else S.INK2, annotation_clip=False)
            ax.set_xlabel('epoch')
            ax.set_ylabel('Mean FG IoU (test)')
            ax.set_ylim(0, 0.42)
            ygrid(ax)
            return None
        return draw
    figure(FIG, 'exploratory_runs', [('sam', 'SAM output', panel(1, 2)), ('generator', 'Generator mask', panel(3, 4))],
           size=(3.6, 2.5))


# ------------------------------------------------------------
# 7  MAPS
# ------------------------------------------------------------
def draw_map(ax, gt, pred=None, boxes=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lons = np.linspace(0, 360, gt.shape[1], endpoint=False)
    lats = np.linspace(-90, 90, gt.shape[0])
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=S.LAND, edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor=S.COAST)
    for cls, label in (('TC', 1), ('AR', 2)):
        ax.contourf(lons, lats, (gt == label).astype(float), levels=[0.5, 1.5], colors=[S.CLASS[cls]], alpha=0.35,
                    transform=ccrs.PlateCarree())
        if pred is not None and pred.get(cls) is not None and pred[cls].any():
            ax.contour(lons, lats, pred[cls].astype(float), levels=[0.5], colors=[S.CLASS_LINE[cls]], linewidths=0.7,
                       transform=ccrs.PlateCarree())
    for bx in boxes or []:
        for x1, y1, x2, y2 in bx:
            ax.add_patch(Rectangle((x1 / 1152 * 360, y1 / 768 * 180 - 90), (x2 - x1) / 1152 * 360, (y2 - y1) / 768 * 180,
                                   fill=False, lw=0.5, ec=S.INK, transform=ccrs.PlateCarree()))
    ax.spines['geo'].set_edgecolor(S.AXIS)
    ax.spines['geo'].set_linewidth(0.6)


def fig_maps(images=(17, 45)):
    import cartopy.crs as ccrs
    ev_dir = os.path.join(RESULTS, 'eval', ENC)
    z = {k: np.load(os.path.join(ev_dir, f'{k}_examples.npz')) for k in ('cgnet_finetuned', 'mpg_seg_s0', 'msf_seg_s0',
                                                                        'learned_prompt_k4_s0')}
    handles = [(Patch(color=S.TC, alpha=0.35), 'TC ground truth'), (Patch(color=S.AR, alpha=0.35), 'AR ground truth'),
               (Line2D([], [], color=S.TC_LINE, lw=1), 'TC prediction'), (Line2D([], [], color=S.AR_LINE, lw=1), 'AR prediction'),
               (Rectangle((0, 0), 1, 1, fill=False, ec=S.INK, lw=0.6), 'box prompt')]
    for img in images:
        get = lambda k, key: z[k][f'{img}__{key}'] if f'{img}__{key}' in z[k].files else None
        gt = get('mpg_seg_s0', 'gt')

        def panel(src, kind=None):
            def draw(ax):
                if src is None:
                    draw_map(ax, gt)
                elif kind is None:
                    draw_map(ax, gt, {c: get(src, f'own_{c}') for c in ('TC', 'AR')})
                elif kind == 'static':
                    draw_map(ax, gt, {c: get(src, f'sam_static_{c}') for c in ('TC', 'AR')})
                else:
                    boxes = [get(src, f'boxes_{kind}_{c}') for c in ('TC', 'AR') if get(src, f'boxes_{kind}_{c}') is not None]
                    draw_map(ax, gt, {c: get(src, f'sam_{kind}_{c}') for c in ('TC', 'AR')}, boxes)
                return handles
            return draw
        panels = [('gt', 'Ground truth', panel(None)),
                  ('cgnet', 'CG-Net (fine-tuned) mask', panel('cgnet_finetuned')),
                  ('msf', 'Multi-scale fusion mask', panel('msf_seg_s0')),
                  ('mpg', 'Mask-prompt generator mask', panel('mpg_seg_s0')),
                  ('mpg_sam_box', 'SAM + box (from MPG)', panel('mpg_seg_s0', 'bbox')),
                  ('mpg_sam_hybrid', 'SAM + hybrid (from MPG)', panel('mpg_seg_s0', 'hybrid')),
                  ('cgnet_sam_box', 'SAM + box (from CG-Net)', panel('cgnet_finetuned', 'bbox')),
                  ('static', 'SAM + learned static prompts', panel('learned_prompt_k4_s0', 'static'))]
        figure(FIG, f'maps_test_image_{img}', panels, ncols=2, size=(3.3, 1.85), legend_ncol=5,
               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})



def fig_robust_decoder():
    """Prompt-robust decoders: corrupted ground-truth prompts and real MPG prompts (mean FG IoU, seeds averaged)."""
    decs = [('phase1', 'Phase-1 decoder', S.INK2, ['']),
            ('box', 'robust decoder, trained on box prompts', S.OUTPUT['sam_bbox'], ['@robust_decoder_box_s0', '@robust_decoder_box_s1']),
            ('hybrid', 'robust decoder, trained on hybrid prompts', S.OUTPUT['sam_hybrid'],
             ['@robust_decoder_hybrid_s0', '@robust_decoder_hybrid_s1'])]
    sweeps = {}
    for key, _, _, tags in decs:
        fs = [os.path.join(RESULTS, '01_oracle_prompts', f'oracle_sweep_{ENC}{t}.csv') for t in tags]
        if not all(os.path.exists(f) for f in fs):
            return
        sweeps[key] = pd.concat([pd.read_csv(f) for f in fs]).groupby('prompt')['Mean FG IoU'].mean()
    handles = lambda: [(Line2D([], [], marker='o', color=c), l) for _, l, c, _ in decs]

    def scale(ax):
        rs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
        for key, _, c, _ in decs:
            ax.plot(rs, [sweeps[key][f'box enlarge {r:+.1f}'] for r in rs], marker='o', color=c)
        ax.axvline(0, color=S.AXIS, lw=0.8)
        ax.set_xlabel('ground-truth box enlarged by (fraction of its size)')
        ax.set_ylabel('mean FG IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return handles()

    def objects_(ax):
        cats = [('box enlarge +0.0', 'none'), ('box drop 25% objects', '25% not\nprompted'),
                ('box + 1 false boxes', '+1 false\nbox'), ('box + 3 false boxes', '+3 false\nboxes'),
                ('box + union logits +-10', 'box +\nmask')]
        x = np.arange(len(cats))
        for j, (key, _, c, _) in enumerate(decs):
            ax.bar(x + (j - 1) * 0.27, [sweeps[key][p] for p, _ in cats], 0.25, color=c, edgecolor='white', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in cats])
        ax.set_ylabel('mean FG IoU (test)')
        ax.set_ylim(0, 1)
        ygrid(ax)
        return handles()

    def real(ax):
        def fg(ev_dir, out):
            v = [r['Mean FG IoU'] for f in sorted(glob.glob(os.path.join(ev_dir, 'mpg_seg_s[0-9].json')))
                 for r in json.load(open(f))['rows'] if r['output'] == out]
            return np.mean(v)
        base = os.path.join(RESULTS, 'eval', ENC)
        dirs = {'phase1': [base], 'box': [os.path.join(base, 'decoder', f'robust_decoder_box_s{i}') for i in (0, 1)],
                'hybrid': [os.path.join(base, 'decoder', f'robust_decoder_hybrid_s{i}') for i in (0, 1)]}
        outs = [('sam_bbox', 'MPG boxes'), ('sam_hybrid', 'MPG hybrid prompts')]
        x = np.arange(len(outs))
        for j, (key, _, c, _) in enumerate(decs):  # dots, not bars: the axis does not start at zero
            ax.plot(x + (j - 1) * 0.2, [np.mean([fg(d, o) for d in dirs[key]]) for o, _ in outs], 'o', ms=7, color=c)
        own = fg(base, 'own')
        ax.axhline(own, color=S.OUTPUT['own'], lw=1.2, ls='--')
        ax.annotate(f'MPG mask {own:.3f}', (x[-1] + 0.45, own), xytext=(0, 3), textcoords='offset points', ha='right',
                    va='bottom', fontsize=7, color=S.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in outs])
        ax.set_xlim(-0.5, len(outs) - 0.5)
        ax.set_ylabel('mean FG IoU (test)')
        ax.set_ylim(0.30, 0.40)
        ygrid(ax)
        return handles()
    figure(FIG, 'robust_decoder', [('box_size', 'Ground-truth boxes of wrong size', scale),
                                   ('objects', 'Missed / false objects', objects_),
                                   ('real_prompts', 'Prompts of the mask-prompt generator', real)], size=(3.3, 2.6))


def main():
    for f in glob.glob(os.path.join(FIG, '*.png')) + glob.glob(os.path.join(FIG, '*.pdf')):
        if not os.path.basename(f).startswith('diagram_'):
            os.remove(f)  # old, differently styled figures
    fig_oracle()
    fig_table413()
    fig_prompters()
    fig_sam_minus_prompter()
    fig_objects()
    fig_training()
    fig_efficiency()
    fig_second_encoder()
    fig_decoder()
    fig_robust_decoder()
    fig_yolo()
    fig_exploratory()
    fig_maps()


if __name__ == '__main__':
    main()
