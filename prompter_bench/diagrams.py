"""Architecture / pipeline diagrams for Section 4.3 (vector PDF + PNG in results/figures)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from common import RESULTS

FIGS = os.path.join(RESULTS, 'figures')
FROZEN, TRAIN, DATA, PROMPT, OUT = '#d9e8f5', '#fde2c8', '#eeeeee', '#e3f1dc', '#f4d9e8'


def box(ax, x, y, w, h, text, color, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=color, ec='#333333', lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', wrap=True)
    return (x, y, w, h)


def arrow(ax, a, b, text='', color='#333333', style='-|>', ls='-', fs=7, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=color, lw=1.0, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha='center', fontsize=fs, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f'{name}.png'), dpi=220, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


def legend(ax, x, y):
    for i, (c, t) in enumerate([(FROZEN, 'frozen (Phase 1)'), (TRAIN, 'trained in Phase 2'), (PROMPT, 'prompt conversion (no weights)')]):
        box(ax, x + i * 2.6, y, 0.35, 0.25, '', c)
        ax.text(x + i * 2.6 + 0.45, y + 0.12, t, va='center', fontsize=7)


def pipeline():
    """Benchmark protocol: every prompter plugs into the same frozen ClimateSAM."""
    fig, ax = canvas(14, 5.6)
    box(ax, 0.2, 2.3, 1.5, 0.9, '16 climate\nvariables\n768x1152', DATA)
    box(ax, 2.1, 2.3, 1.6, 0.9, 'Input adapter\n+ ViT-B encoder\n(Infused Token)', FROZEN)
    box(ax, 4.2, 3.4, 2.0, 0.9, 'Image embedding\n256x64x64\n+ 12 ViT blocks', DATA)
    box(ax, 4.2, 0.6, 2.0, 0.9, 'TMQ, U850, V850, PSL\n(raw, 4 channels)', DATA)
    arrow(ax, (1.7, 2.75), (2.1, 2.75))
    arrow(ax, (3.7, 2.9), (4.2, 3.8))
    arrow(ax, (1.0, 2.3), (4.2, 1.05), rad=0.25)
    box(ax, 6.7, 3.2, 2.2, 1.3, 'SAM-feature prompter\nlogistic regression /\nmulti-scale fusion (+token) /\nmask-prompt generator', TRAIN, fs=7)
    box(ax, 6.7, 0.5, 2.2, 1.1, 'External prompter\nCG-Net (segmentation)\nCG-Net + YOLO head', TRAIN, fs=7)
    arrow(ax, (6.2, 3.85), (6.7, 3.85))
    arrow(ax, (6.2, 1.05), (6.7, 1.05))
    box(ax, 9.4, 2.0, 1.7, 1.5, 'TC / AR maps\n(logits)\n\n"prompter mask"', OUT, fs=8)
    arrow(ax, (8.9, 3.85), (9.4, 3.2))
    arrow(ax, (8.9, 1.05), (9.4, 2.3))
    box(ax, 11.5, 2.0, 1.4, 1.5, 'Prompt\nconversion\nbox / points /\nmask logits /\nhybrid', PROMPT, fs=7)
    arrow(ax, (11.1, 2.75), (11.5, 2.75))
    box(ax, 11.2, 4.1, 2.5, 0.8, 'Prompt encoder +\nHQ mask decoder', FROZEN)
    arrow(ax, (12.2, 3.5), (12.4, 4.1))
    arrow(ax, (5.2, 4.3), (11.2, 4.6), rad=-0.25)
    ax.text(8.3, 5.35, 'image embedding + block-1 features', ha='center', fontsize=6)
    box(ax, 13.0, 2.2, 0.9, 1.1, 'SAM\nmask', OUT)
    arrow(ax, (13.4, 4.1), (13.45, 3.3))
    ax.text(10.25, 1.55, 'evaluated\nwithout SAM', ha='center', fontsize=6, style='italic')
    ax.text(13.45, 1.9, 'evaluated\nwith SAM', ha='center', fontsize=6, style='italic')
    legend(ax, 0.2, 4.7)
    save(fig, 'diagram_benchmark_pipeline')


def mask_prompt_generator():
    """Architecture of the mask-prompt generator and how its maps become SAM prompts."""
    fig, ax = canvas(14, 5.6)
    box(ax, 0.2, 4.1, 1.9, 0.8, 'Image embedding\n256x64x64', DATA)
    box(ax, 0.2, 2.9, 1.9, 0.8, 'ViT block 6\n768x64x64', DATA)
    box(ax, 0.2, 1.7, 1.9, 0.8, 'ViT block 12\n768x64x64', DATA)
    box(ax, 2.6, 4.1, 1.5, 0.8, '1x1 conv\n256 -> 128', TRAIN)
    box(ax, 2.6, 2.3, 1.5, 0.8, 'concat + 1x1 conv\n1536 -> 128', TRAIN, fs=7)
    arrow(ax, (2.1, 4.5), (2.6, 4.5))
    arrow(ax, (2.1, 3.3), (2.6, 2.8))
    arrow(ax, (2.1, 2.1), (2.6, 2.6))
    ax.text(4.55, 3.55, '+', fontsize=16, ha='center', va='center')
    arrow(ax, (4.1, 4.5), (4.45, 3.7))
    arrow(ax, (4.1, 2.7), (4.45, 3.4))
    box(ax, 4.9, 2.9, 2.1, 1.3, '3 ConvNeXt blocks\n(dw 3x3, dilation 1/2/4,\nLayerNorm, MLP x4)\n128x64x64', TRAIN, fs=7)
    arrow(ax, (4.7, 3.55), (4.9, 3.55))
    box(ax, 7.4, 2.9, 1.9, 1.3, '2x ConvTranspose\n(x2 each)\n64 -> 128 -> 256\n32x256x256', TRAIN, fs=7)
    arrow(ax, (7.0, 3.55), (7.4, 3.55))
    box(ax, 9.7, 2.9, 1.3, 1.3, '3x3 conv\n-> 2 logits\n(TC, AR)\n256x256', TRAIN, fs=7)
    arrow(ax, (9.3, 3.55), (9.7, 3.55))
    box(ax, 11.5, 4.1, 2.3, 0.9, 'AR: logit map as\ndense mask prompt', PROMPT, fs=7)
    box(ax, 11.3, 2.2, 1.95, 1.2, 'TC: one box per blob\n(connected components,\n<= 16) + TC logit map\nas dense mask prompt', PROMPT, fs=7)
    arrow(ax, (11.0, 3.8), (11.5, 4.5))
    arrow(ax, (11.0, 3.3), (11.5, 2.8))
    box(ax, 11.5, 0.5, 2.3, 1.0, 'frozen prompt encoder\n+ HQ decoder -> SAM masks', FROZEN, fs=7)
    arrow(ax, (12.3, 2.2), (12.3, 1.5))
    arrow(ax, (13.5, 4.1), (13.5, 1.5))
    ax.text(6.0, 1.3, 'bias of the head initialised to a 5% foreground prior -> initial prompts are "empty"\n'
                      'differentiable path: logits -> mask downscaling -> decoder (box coordinates are not differentiated)',
            ha='center', fontsize=7, style='italic')
    ax.text(6.0, 5.2, 'Mask-prompt generator: 671k parameters, all computation at 64x64 or below until the last two layers',
            ha='center', fontsize=9, weight='bold')
    legend(ax, 0.2, 0.3)
    save(fig, 'diagram_mask_prompt_generator')


def training_modes():
    """Where the loss is applied in the three training modes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    titles = ['(a) segmentation loss', '(b) end-to-end through SAM', '(c) two-stage: (a), then (b) at lr/10']
    for ax, title, sam_loss in zip(axes, titles, (False, True, True)):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 3.5)
        ax.axis('off')
        ax.set_title(title, fontsize=10)
        box(ax, 0.1, 1.2, 1.1, 0.9, 'frozen\nfeatures', FROZEN, fs=7)
        box(ax, 1.5, 1.2, 1.1, 0.9, 'prompter', TRAIN, fs=8)
        box(ax, 3.0, 1.2, 1.0, 0.9, 'TC/AR\nlogits', OUT, fs=7)
        arrow(ax, (1.2, 1.65), (1.5, 1.65))
        arrow(ax, (2.6, 1.65), (3.0, 1.65))
        box(ax, 3.0, 2.6, 1.0, 0.6, 'loss', '#ffffff', fs=7)
        arrow(ax, (3.5, 2.1), (3.5, 2.6))
        if sam_loss:
            box(ax, 3.0, 0.1, 1.9, 0.7, 'frozen SAM decoder', FROZEN, fs=7)
            arrow(ax, (3.5, 1.2), (3.5, 0.8))
            box(ax, 4.2, 1.2, 0.75, 0.9, 'loss', '#ffffff', fs=7)
            arrow(ax, (4.6, 0.8), (4.6, 1.2))
            arrow(ax, (4.6, 1.2), (2.0, 1.2), color='#c0392b', ls='--', rad=0.35)
            ax.text(3.3, 0.95, 'gradient', fontsize=6, color='#c0392b')
    save(fig, 'diagram_training_modes')


if __name__ == '__main__':
    pipeline()
    mask_prompt_generator()
    training_modes()
