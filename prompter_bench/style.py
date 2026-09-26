"""
One visual grammar for every figure of Section 4.3 (print, light background).

Colour always means the same thing:
  classes        TC green, AR blue (as in thesis Figure 2.2); on maps: fill = ground truth, outline = prediction
  final output   prompter mask = neutral grey; SAM + box = orange; SAM + hybrid = violet; SAM + points = yellow;
                 SAM + learned static prompts = magenta; SAM with the adapted decoder = aqua
  emphasis       the proposed mask-prompt generator = red, every other method = grey (+ direct label)
Palettes validated with the dataviz validator (class pair: all checks pass; output set: passes in this order,
contrast relief via legends and tables). Box prompts on maps are thin black rectangles (orange on green fails CVD).

Every multi-panel figure is also saved panel by panel (figures/panels/<name>_<panel>.{png,pdf}).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
TC, AR = '#008300', '#2a78d6'
TC_LINE, AR_LINE = '#005a00', '#184f95'
CLASS = {'TC': TC, 'AR': AR}
CLASS_LINE = {'TC': TC_LINE, 'AR': AR_LINE}
OUTPUT = {
    'own': '#8a8881',
    'sam_bbox': '#eb6834',
    'sam_hybrid': '#4a3aa7',
    'sam_point': '#eda100',
    'sam_static': '#e87ba4',
    'adapted': '#1baf7a',
}
EMPH, OTHER = '#e34948', '#a3a19a'
LAND, COAST = '#efefec', '#898781'

PANEL = (3.3, 2.5)  # inches, one panel (half of a 16 cm text width)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'lines.linewidth': 1.5, 'lines.markersize': 4, 'errorbar.capsize': 2,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
})


# ordinal 3-step ramps per class (validated: monotone, visible steps, light end >= 2:1 on white)
RAMP3 = {'TC': ['#7cc47c', '#008300', '#004000'], 'AR': ['#86b6ef', '#2a78d6', '#104281']}


def ygrid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def figure(out_dir, name, panels, ncols=None, size=PANEL, legend_ncol=None, subplot_kw=None, sharey=False, sharex=False):
    """
    panels: list of (key, title, draw) with draw(ax) -> list of (handle, label) for the legend (or None).
    Saves <out_dir>/<name> (all panels, shared legend on top) and <out_dir>/panels/<name>_<key> (one per panel).
    """
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False,
                             subplot_kw=subplot_kw or {}, sharey=sharey, sharex=sharex)
    handles = []
    for i, (key, title, draw) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        h = draw(ax) or []
        handles = handles or h
        if title:
            ax.set_title(f'({chr(97 + i)}) {title}' if n > 1 else title)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    if handles:
        fig.legend(*zip(*handles), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=legend_ncol or len(handles),
                   frameon=False)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, name))
    if n > 1:
        for key, title, draw in panels:
            f, ax = plt.subplots(figsize=size, subplot_kw=subplot_kw or {})
            h = draw(ax) or []
            if title:
                ax.set_title(title)
            if h:  # above the axes, as in the combined figure: never covers data
                short = sum(len(l) for _, l in h) <= 45
                f.legend(*zip(*h), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(h) if short else 1,
                         frameon=False)
            f.tight_layout()
            save(f, os.path.join(out_dir, 'panels', f'{name}_{key}'))
