"""
Tables and figures of results/08_thesis_recheck (re-evaluation of the thesis' own results).
Inputs: phase1_*.csv (phase1_recheck.py), phase1_examples.npz (phase1_examples.py), wandb_histories.json (the author's
training logs), eval/<encoder>/user_*.json (phase2_recheck.py), param_counts.json, input_adapter_weights.json.
"""
import os
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
from figures import draw_map, label_ends

OUT = os.path.join(RESULTS, '08_thesis_recheck')
FIG = os.path.join(RESULTS, 'figures')
REPORTED, ORIGINAL, FIXED = S.INK, S.OUTPUT['own'], S.EMPH  # thesis value, re-run as in the thesis, re-run corrected

PHASE1_ROWS = ['Infused Token, Linear, 1.0', 'Infused Token, Linear, 0.5', 'Infused Token, Nonlinear, 1.0',
               'Infused Token, Nonlinear, 0.5']
SHORT1 = {'Infused Token, Linear, 1.0': 'linear, MLP 1.0', 'Infused Token, Linear, 0.5': 'linear, MLP 0.5',
          'Infused Token, Nonlinear, 1.0': 'nonlinear, MLP 1.0', 'Infused Token, Nonlinear, 0.5': 'nonlinear, MLP 0.5'}
PROMPT_NAME = {'bbox': 'box', 'point': 'point', 'random': 'random'}


# ------------------------------------------------------------------------------------------------------------------
# PHASE 1
# ------------------------------------------------------------------------------------------------------------------
def phase1_frame():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'phase1_*.csv')))], ignore_index=True)
    df['thesis_row'] = df['thesis_row'].fillna('')
    return df


def phase1_table(df):
    body = []
    for row in PHASE1_ROWS:
        d = df[df['thesis_row'] == row]
        first = True
        for p in ('bbox', 'point', 'random'):
            e = d[d['prompt'] == p]
            if e.empty:
                continue
            th = (e['thesis TC IoU'].iloc[0], e['thesis AR IoU'].iloc[0])
            cells = []
            for path in ('original', 'fixed'):
                g = e[e['path'] == path]
                if p == 'random':
                    cells += [f"{g['TC IoU'].mean():.3f} ({g['TC IoU'].min():.3f}--{g['TC IoU'].max():.3f})",
                              f"{g['AR IoU'].mean():.3f} ({g['AR IoU'].min():.3f}--{g['AR IoU'].max():.3f})"]
                else:
                    cells += [f"{g['TC IoU'].iloc[0]:.3f}", f"{g['AR IoU'].iloc[0]:.3f}"]
            fx = e[e['path'] == 'fixed']
            delta = (fx['TC IoU'].mean() + fx['AR IoU'].mean()) / 2 - (th[0] + th[1]) / 2
            body.append([SHORT1[row] if first else '', PROMPT_NAME[p], f'{th[0]:.3f}', f'{th[1]:.3f}'] + cells + [f'{delta:+.3f}'])
            first = False
        body.append('MIDRULE')
    mr.write_table('thesis_recheck_phase1',
                   ['Infused Token', 'Prompt', 'Thesis TC', 'Thesis AR', 'As in thesis TC', 'As in thesis AR',
                    'Adapter applied TC', 'Adapter applied AR', '$\\Delta$ FG (applied $-$ thesis)'], body[:-1],
                   'Re-evaluation of the Phase-1 checkpoints (appendix Table 1) on the 61 test images with the thesis '
                   'pipeline. "As in thesis": images encoded with set\\_infer\\_img (first three raw channels, learned input '
                   'adapter skipped), as in the validation of train\\_adaptation.py; "adapter applied": encode\\_images, as '
                   'in training. Random prompts: mean of three prompt draws (range).', 'tab:thesis_recheck_phase1',
                   'llccccccc')


def fig_phase1(df):
    rows = [(r, p) for r in PHASE1_ROWS for p in ('bbox', 'point', 'random')]

    def panel(cls):
        def draw(ax):
            for i, (r, p) in enumerate(rows):
                e = df[(df['thesis_row'] == r) & (df['prompt'] == p)]
                if e.empty:
                    continue
                th = e[f'thesis {cls} IoU'].iloc[0]
                for path, col, dy in (('original', ORIGINAL, -0.12), ('fixed', FIXED, 0.12)):
                    v = e[e['path'] == path][f'{cls} IoU']
                    ax.plot([v.min(), v.max()], [i + dy] * 2, color=col, lw=1.2, solid_capstyle='round')
                    ax.plot(v.mean(), i + dy, 'o', color=col, ms=4.5)
                ax.plot(th, i, 'o', mfc='white', mec=REPORTED, mew=1.1, ms=5.5, zorder=4)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f'{SHORT1[r]}, {PROMPT_NAME[p]}' for r, p in rows])
            ax.set_ylim(len(rows) - 0.5, -0.5)
            for k in (3, 6, 9):
                ax.axhline(k - 0.5, color=S.GRID, lw=0.8)
            ax.set_xlabel(f'{cls} IoU (test)')
            ygrid(ax, 'x')
            return [(Line2D([], [], marker='o', ls='', mfc='white', mec=REPORTED, mew=1.1), 'reported in the thesis'),
                    (Line2D([], [], marker='o', ls='', color=ORIGINAL), 're-run as in the thesis (adapter skipped)'),
                    (Line2D([], [], marker='o', ls='', color=FIXED), 're-run with the input adapter applied')]
        return draw
    figure(FIG, 'thesis_recheck_phase1', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR'))], size=(3.3, 3.6),
           sharey=True, legend_ncol=3)


def adapter_bootstrap():
    """Paired bootstrap (per-image counts, 2000 resamples): effect of the skipped adapter and linear vs. nonlinear."""
    npz = lambda tag: np.load(os.path.join(OUT, f'phase1_{tag}_per_image.npz'))
    pairs = [('1.0', 'infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 'infused_token_vit_b_1.0_infused_token_vit_b_mlp1_nonlinear'),
             ('0.5', 'infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED', 'infused_token_vit_b_0.5_infused_token_vit_b_mlp05_nonlinear')]
    fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]"
    body, rows = [], []
    for mlp, lin, non in pairs:
        L, N = npz(lin), npz(non)
        for p in ('bbox_s0', 'point_s0', 'random_s0'):
            comps = [('nonlinear: adapter applied vs. skipped', N[f'{p}_original'], N[f'{p}_fixed']),
                     ('linear vs. nonlinear, as in the thesis', N[f'{p}_original'], L[f'{p}_original']),
                     ('linear vs. nonlinear, adapter applied', N[f'{p}_fixed'], L[f'{p}_fixed'])]
            for label, a, b in comps:
                r = mr.bootstrap(a.astype(float), b.astype(float))
                body.append([f'MLP {mlp}', PROMPT_NAME[p.split('_')[0]], label, fmt(r['TC']), fmt(r['AR']), fmt(r['FG'])])
                rows.append({'mlp': mlp, 'prompt': p, 'comparison': label, **{f'{k} {j}': float(v[i]) for k, v in r.items()
                                                                            for i, j in enumerate(('delta', 'lo', 'hi', 'p_gt0'))}})
            body.append('MIDRULE')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'adapter_bootstrap.csv'), index=False)
    mr.write_table('thesis_recheck_adapter_bootstrap', ['Infused Token', 'Prompt', 'Comparison (B $-$ A)', '$\\Delta$ TC [95\\% CI]',
                                                        '$\\Delta$ AR [95\\% CI]', '$\\Delta$ mean FG [95\\% CI]'], body[:-1],
                   'Linear versus nonlinear input adapter (Table 4.6) and the effect of the skipped adapter, paired bootstrap '
                   'over the 61 test images (random prompts: first draw, identical for both models).',
                   'tab:thesis_recheck_adapter_bootstrap', 'lllccc')


def fig_adapter_inputs():
    z = np.load(os.path.join(OUT, 'phase1_examples.npz'))
    img = 17
    srcs = [('raw', 'raw', z[f'{img}__raw_rgb']),
            ('linear', 'linear adapter', z[f'{img}__linear_1.0_adapter_rgb']),
            ('nonlinear', 'nonlinear adapter', z[f'{img}__nonlinear_1.0_adapter_rgb'])]
    chans = ['R', 'G', 'B']
    raw_names = ['TMQ', 'U850', 'V850']

    def panel(arr, c):
        def draw(ax):
            ax.imshow(arr[c][::-1], cmap='Greys_r', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(S.AXIS)
            ax.text(0.01, 0.03, f'range {arr[c].min():.0f} to {arr[c].max():.0f}, mean {arr[c].mean():.0f}',
                    transform=ax.transAxes, fontsize=6, color='white', va='bottom')
            return None
        return draw
    panels = [(f'{k}_{chans[c]}', f'{title}, {chans[c]}' + (f' ({raw_names[c]})' if k == 'raw' else ''), panel(arr, c))
              for k, title, arr in srcs for c in range(3)]
    figure(FIG, 'thesis_recheck_adapter_inputs', panels, ncols=3, size=(2.6, 2.0))


def fig_nonlinear_maps():
    import cartopy.crs as ccrs
    z = np.load(os.path.join(OUT, 'phase1_examples.npz'))
    handles = [(Patch(color=S.TC, alpha=0.35), 'TC ground truth'), (Patch(color=S.AR, alpha=0.35), 'AR ground truth'),
               (Line2D([], [], color=S.TC_LINE, lw=1), 'TC prediction'), (Line2D([], [], color=S.AR_LINE, lw=1), 'AR prediction'),
               (Rectangle((0, 0), 1, 1, fill=False, ec=S.INK, lw=0.6), 'box prompt (ground truth)')]
    for img in (17, 45):
        gt = z[f'{img}__gt']
        boxes = [z[f'{img}__boxes_{c}'] for c in ('TC', 'AR')]

        def panel(key):
            def draw(ax):
                if key is None:
                    draw_map(ax, gt, None, boxes)
                else:
                    draw_map(ax, gt, {c: z[f'{img}__{key}_{c}'] for c in ('TC', 'AR')}, boxes)
                return handles
            return draw
        panels = [('gt', 'Ground truth and box prompts', panel(None)),
                  ('linear', 'Linear, as in the thesis', panel('linear_1.0_original')),
                  ('nonlinear_thesis', 'Nonlinear, adapter skipped (thesis)', panel('nonlinear_1.0_original')),
                  ('nonlinear_fixed', 'Nonlinear, adapter applied', panel('nonlinear_1.0_fixed'))]
        figure(FIG, f'thesis_recheck_maps_nonlinear_{img}', panels, ncols=2, size=(3.3, 1.85), legend_ncol=5,
               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})


# ------------------------------------------------------------------------------------------------------------------
# TRAINING LOGS (author's wandb runs)
# ------------------------------------------------------------------------------------------------------------------
LOG_RUNS = [  # run name, label, thesis row, colour
    ('infused_token_vit_b_mlp1_CORRECTED', 'Infused Token 1.0', S.OTHER),
    ('infused_token_vit_b_mlp05_CORRECTED', 'Infused Token 0.5', S.OTHER),
    ('single_sam_vit_b_CORRECTED', 'Single Token 1.0', S.OTHER),
    ('lora_single_vit_b_r32_CORRECTED', 'Single LoRA r32', S.EMPH),
    ('lora_single_vit_b_r64_CORRECTED', 'Single LoRA r64', '#a8201a'),
]


def log_runs():
    h = json.load(open(os.path.join(OUT, 'wandb_histories.json')))
    out = {}
    for name, runs in h.items():
        best = max(runs, key=lambda r: len([x for x in r['history'] if 'valid/miou_tc' in x]))  # the complete run
        rows = [x for x in best['history'] if isinstance(x.get('valid/miou_tc'), (int, float))]
        out[name] = {'dir': best['dir'], 'epoch': [x.get('epoch') for x in rows], 'tc': [x['valid/miou_tc'] for x in rows],
                     'ar': [x['valid/miou_ar'] for x in rows],
                     'gen_tc': [x.get('valid/logit_tc_iou') for x in rows], 'gen_ar': [x.get('valid/logit_ar_iou') for x in rows]}
    return out


def logs_table(logs):
    body = []
    names = [n for n, _, _ in LOG_RUNS] + ['infused_token_vit_b_mlp1_nonlinear', 'infused_token_vit_b_mlp05_nonlinear',
                                            'single_sam_vit_b_nonlinear', 'lora_single_vit_b_r32_nonlinear',
                                            'lora_single_vit_b_r64_nonlinear', 'infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH']
    for n in names:
        if n not in logs:
            continue
        L = logs[n]
        mean = [(a + b) / 2 for a, b in zip(L['tc'], L['ar'])]
        i = int(np.argmax(mean))
        itc, iar = int(np.argmax(L['tc'])), int(np.argmax(L['ar']))
        body.append([n.replace('_', '\\_'), f"{max(L['tc']):.3f} (ep. {L['epoch'][itc]})", f"{max(L['ar']):.3f} (ep. {L['epoch'][iar]})",
                     f"{L['tc'][i]:.3f} / {L['ar'][i]:.3f} (ep. {L['epoch'][i]})", f"{L['tc'][-1]:.3f} / {L['ar'][-1]:.3f} (ep. {L['epoch'][-1]})"])
    mr.write_table('thesis_recheck_training_logs', ['Run (wandb)', 'Max TC IoU', 'Max AR IoU', 'Best mean epoch TC / AR',
                                                    'Last epoch TC / AR'], body,
                   'Test-set IoU logged during Phase-1 training (the author\'s wandb runs; validation = the 61 test images, '
                   'every 5 epochs, input adapter skipped). The saved checkpoint is the epoch with the best mean of TC and AR.',
                   'tab:thesis_recheck_training_logs', 'lcccc')


def fig_training_logs(logs):
    def panel(key, cls):
        def draw(ax):
            ends = []
            for n, label, col in LOG_RUNS:
                if n not in logs:
                    continue
                L = logs[n]
                ax.plot(L['epoch'], L[key], color=col, lw=1.5, marker='o', ms=2.5)
                mean = [(a + b) / 2 for a, b in zip(L['tc'], L['ar'])]
                i = int(np.argmax(mean))
                ax.plot(L['epoch'][i], L[key][i], 'o', mfc='white', mec=col, mew=1.2, ms=6, zorder=5)
                ends.append((L[key][-1], label, S.INK2 if col == S.OTHER else col, L['epoch'][-1] + 1.5))
            label_ends(ax, ends, None, 0.043)
            ax.set_xlim(0, 64)
            ax.set_ylim(0, 0.75)
            ax.set_xlabel('epoch')
            ax.set_ylabel(f'{cls} IoU on the test set (logged)')
            ygrid(ax)
            return [(Line2D([], [], color=S.OTHER, lw=1.5), 'token-based adapters'), (Line2D([], [], color=S.EMPH, lw=1.5), 'Single LoRA'),
                    (Line2D([], [], marker='o', ls='', mfc='white', mec=S.INK2), 'saved checkpoint (best mean on the test set)')]
        return draw
    figure(FIG, 'thesis_recheck_training_logs', [('tc', 'TC', panel('tc', 'TC')), ('ar', 'AR', panel('ar', 'AR'))],
           size=(3.4, 2.6), legend_ncol=3)


# ------------------------------------------------------------------------------------------------------------------
# PHASE 2
# ------------------------------------------------------------------------------------------------------------------
USER = [  # key, wandb run, label, encoder tag, prompt type used in training, retrained counterpart (method key)
    ('user_msf_Generator', 'Generator', 'MSF (Generator)', 'infused_mlp1_best', 'point', 'msf_seg'),
    ('user_msf_bbox_128', 'generator_128_vit_b_bbox', 'MSF (generator_128_vit_b_bbox)', 'infused_05_retrain', 'bbox', 'msf_seg'),
    ('user_token_FIRST', 'FIRST', 'MSF + token gate (FIRST)', 'infused_mlp1_best', 'point', 'msf_token_seg'),
    ('user_token_cg_CG', 'CG', 'token gate + CG blocks (CG)', 'infused_mlp1_best', 'point', 'msf_token_cg_seg'),
    ('user_token_cg_CG_256', 'CG_256', 'token gate + CG blocks (CG_256)', 'infused_mlp1_best', 'bbox', 'msf_token_cg_seg'),
    ('user_token_cg_CG_256_REAL', 'CG_256_REAL', 'token gate + CG blocks (CG_256_REAL)', 'infused_mlp1_best', 'bbox', 'msf_token_cg_seg'),
    ('user_sp_token_SP_TOKEN', 'SP_TOKEN', 'token gate, shared weights (SP_TOKEN)', 'infused_mlp1_best', 'bbox', 'msf_sp_token_seg'),
    ('user_logreg_Logistic', 'Logistic', 'logistic regression, block 1 (Logistic)', 'infused_mlp1_best', 'bbox', 'logreg_l0_seg'),
]


def phase2_frame(logs):
    ev, _ = mr.load_eval('infused_mlp1')
    mine = ev.groupby(['method', 'output'])[['TC IoU', 'AR IoU', 'Mean FG IoU']].mean()
    rows = []
    for key, run, label, enc, pt, counterpart in USER:
        f = os.path.join(RESULTS, 'eval', enc, f'{key}.json')
        if not os.path.exists(f):
            continue
        d = {r['output']: r for r in json.load(open(f))['rows']}
        L = logs.get(run)
        r = {'key': key, 'label': label, 'encoder': enc, 'prompt': pt,
             'own TC': d['own']['TC IoU'], 'own AR': d['own']['AR IoU'],
             'sam TC': d[f'sam_{"bbox" if pt == "bbox" else "point"}']['TC IoU'],
             'sam AR': d[f'sam_{"bbox" if pt == "bbox" else "point"}']['AR IoU'],
             'hybrid TC': d['sam_hybrid']['TC IoU'], 'hybrid AR': d['sam_hybrid']['AR IoU'],
             'AR objects / image': d['own'].get('AR objects', np.nan) / 61}
        if L and any(v is not None for v in L['gen_tc']):
            gt_, ga = [v for v in L['gen_tc'] if v is not None], [v for v in L['gen_ar'] if v is not None]
            r.update({'logged own TC (best)': max(gt_), 'logged own AR (best)': max(ga), 'logged own TC (last)': gt_[-1],
                      'logged own AR (last)': ga[-1], 'logged SAM TC (best)': max(L['tc']), 'logged SAM AR (best)': max(L['ar']),
                      'logged epochs': len(L['tc'])})
        if (counterpart, 'own') in mine.index:
            r['retrained TC'], r['retrained AR'] = mine.loc[(counterpart, 'own'), 'TC IoU'], mine.loc[(counterpart, 'own'), 'AR IoU']
        rows.append(r)
    return pd.DataFrame(rows)


def phase2_table(p2):
    f = lambda v: '--' if pd.isna(v) else f'{v:.3f}'
    body = [[r['label'].replace('_', '\\_'), r['prompt'].replace('bbox', 'box'), f"{f(r.get('logged own TC (best)'))} / {f(r.get('logged own AR (best)'))}",
             f"{f(r['own TC'])} / {f(r['own AR'])}", f"{f(r.get('logged SAM TC (best)'))} / {f(r.get('logged SAM AR (best)'))}",
             f"{f(r['sam TC'])} / {f(r['sam AR'])}", f"{f(r.get('retrained TC'))} / {f(r.get('retrained AR'))}"]
            for _, r in p2.iterrows()]
    mr.write_table('thesis_recheck_phase2', ['Prompter checkpoint (run)', 'Prompts', 'Logged mask (best)', 'Re-run mask',
                                             'Logged SAM (best)', 'Re-run SAM', 'Same arch. retrained'], body,
                   'The author\'s Phase-2 prompter checkpoints (TC IoU / AR IoU, 61 test images), each with the frozen encoder '
                   'it was trained with. Logged: best value in the author\'s wandb run (selected on the test set; the run '
                   'logged SAM with its own prompt pipeline). Re-run mask: 3-class argmax, as in the training scripts. '
                   'Same arch. retrained: the architecture trained under the benchmark protocol (two sigmoid channels, '
                   'selection on validation, primary checkpoint, mean of seeds). SAM hybrid and object counts: 08\\_thesis\\_recheck/phase2\\_all.csv.',
                   'tab:thesis_recheck_phase2', 'llccccc')


def fig_phase2(p2):
    labels = list(p2['label'])

    def panel(cls):
        def draw(ax):
            for i, (_, r) in enumerate(p2.iterrows()):
                if not pd.isna(r.get(f'logged own {cls} (best)', np.nan)):
                    ax.plot(r[f'logged own {cls} (best)'], i, 'o', mfc='white', mec=REPORTED, mew=1.1, ms=5.5, zorder=4)
                ax.plot(r[f'own {cls}'], i, 'o', color=ORIGINAL, ms=5)
                col = S.OUTPUT['sam_bbox'] if r['prompt'] == 'bbox' else S.OUTPUT['sam_point']
                ax.plot(r[f'sam {cls}'], i + 0.22, 's', color=col, ms=3.8)
                if not pd.isna(r.get(f'retrained {cls}', np.nan)):
                    ax.plot(r[f'retrained {cls}'], i - 0.22, 'D', color=FIXED, ms=3.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.set_xlim(0, 0.45)
            ax.set_xlabel(f'{cls} IoU (test)')
            ygrid(ax, 'x')
            return [(Line2D([], [], marker='o', ls='', mfc='white', mec=REPORTED, mew=1.1), 'logged mask (best epoch)'),
                    (Line2D([], [], marker='o', ls='', color=ORIGINAL), 're-run mask'),
                    (Line2D([], [], marker='s', ls='', color=S.OUTPUT['sam_bbox']), 're-run SAM + box'),
                    (Line2D([], [], marker='s', ls='', color=S.OUTPUT['sam_point']), 're-run SAM + points'),
                    (Line2D([], [], marker='D', ls='', color=FIXED), 'same architecture, retrained')]
        return draw
    figure(FIG, 'thesis_recheck_phase2', [('tc', 'TC', panel('TC')), ('ar', 'AR', panel('AR'))], size=(3.3, 3.2),
           sharey=True, legend_ncol=3)


def fig_phase2_maps():
    import cartopy.crs as ccrs
    load = lambda enc, key: np.load(os.path.join(RESULTS, 'eval', enc, f'{key}_examples.npz'))
    # only checkpoints whose training encoder still exists (their re-run reproduces the author's log)
    z = {'user_msf': load('infused_05_retrain', 'user_msf_bbox_128'), 'user_cg': load('infused_mlp1_best', 'user_logreg_Logistic'),
         'msf': load('infused_mlp1', 'msf_seg_s0'), 'mpg': load('infused_mlp1', 'mpg_seg_s0')}
    handles = [(Patch(color=S.TC, alpha=0.35), 'TC ground truth'), (Patch(color=S.AR, alpha=0.35), 'AR ground truth'),
               (Line2D([], [], color=S.TC_LINE, lw=1), 'TC prediction'), (Line2D([], [], color=S.AR_LINE, lw=1), 'AR prediction'),
               (Rectangle((0, 0), 1, 1, fill=False, ec=S.INK, lw=0.6), 'box prompt')]
    for img in (17, 45):
        get = lambda k, key: z[k][f'{img}__{key}'] if f'{img}__{key}' in z[k].files else None
        gt = get('mpg', 'gt')

        def panel(src, kind=None):
            def draw(ax):
                if kind is None:
                    draw_map(ax, gt, {c: get(src, f'own_{c}') for c in ('TC', 'AR')})
                else:
                    boxes = [get(src, f'boxes_{kind}_{c}') for c in ('TC', 'AR') if get(src, f'boxes_{kind}_{c}') is not None]
                    draw_map(ax, gt, {c: get(src, f'sam_{kind}_{c}') for c in ('TC', 'AR')}, boxes)
                return handles
            return draw
        panels = [('user_msf', 'Thesis MSF checkpoint: mask', panel('user_msf')),
                  ('user_msf_sam', 'Thesis MSF checkpoint: SAM + box', panel('user_msf', 'bbox')),
                  ('user_logreg', 'Thesis logistic regression (block 1): mask', panel('user_cg')),
                  ('msf', 'MSF retrained under the protocol: mask', panel('msf')),
                  ('mpg', 'Mask-prompt generator: mask', panel('mpg')),
                  ('mpg_sam', 'Mask-prompt generator: SAM + hybrid', panel('mpg', 'hybrid'))]
        figure(FIG, f'thesis_recheck_maps_prompters_{img}', panels, ncols=2, size=(3.3, 1.85), legend_ncol=5,
               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})


def main():
    df = phase1_frame()
    df.to_csv(os.path.join(OUT, 'phase1_all.csv'), index=False)
    phase1_table(df)
    fig_phase1(df)
    adapter_bootstrap()
    fig_adapter_inputs()
    fig_nonlinear_maps()
    logs = log_runs()
    logs_table(logs)
    fig_training_logs(logs)
    p2 = phase2_frame(logs)
    p2.to_csv(os.path.join(OUT, 'phase2_all.csv'), index=False)
    if not p2.empty:
        phase2_table(p2)
        fig_phase2(p2)
        fig_phase2_maps()


if __name__ == '__main__':
    main()
