'''
Paper figures of task A2 (heterogeneous batch vs isolated systems), one figure per file.

  forces_<box set>_<term>   relative force error per system, one column per comparison
  energy_<box set>_<term>   the same for the potential energy of each system
  nve_<box set>_<engine>    largest atom displacement mixed vs isolated along the NVE run

and figures that put several box sets or engines on the same axes:

  forces_all_<term>         median and worst force error of each comparison, the three box sets
  energy_all_<term>         the same for the potential energy
  nve_max_all               worst system at each step, both box sets against both engines
  nve_species_time          worst system of each species against simulated time
  nve_final                 displacement of every system at the last step

Reads fuerzas_<box set>.npz and nve_<box set>.npz written by the A2 tests.
'''
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Fixed order, one colour and one marker per species
SPECIES = {'cln':    ('Chignolin',            '#2a78d6', 'o'),
           'trp':    ('Trp-cage',             '#eb6834', 's'),
           'bbl':    ('BBL',                  '#1baf7a', '^'),
           'lambda': (r'$\lambda$-repressor', '#eda100', 'D')}
# Same look as fig_rmsd.py, the figures of reporte_resultados
TEXT = '#52514e'
GRID = '#b8b8b3'

MM = 1 / 25.4
WIDTH = 170 * MM          # \linewidth of main.tex
DPI = 300

# Box side in Å of the 'data' set, which keeps the box of each xtc and is not written to the results.
# Rounded: the xtc boxes are cubic to within 0.3 Å
DATA_BOX = {'cln': 40, 'trp': 38, 'bbl': 60, 'lambda': 71}

BOXES = {'data': 'data boxes', 'report': 'large boxes', 'tight': 'tight boxes', 'images': 'boxes with image contacts'}

TERMS = {'priors': 'priors', 'nnp': 'priors + NNP'}
# The energy rows of a term carry this suffix in the npz, and their own tolerance in config.json
ENERGY = '_E'
ENGINES = {'fiumd': 'FiuMD', 'torchmd': 'TorchMD'}

EPS = np.finfo(np.float64).eps

# For the figures that compare box sets or engines instead of species
BOX_STYLE = {'data': ('#2a78d6', 'o'), 'report': ('#1baf7a', 's'), 'tight': ('#eb6834', '^')}
ENGINE_COLORS = {'fiumd': '#2a78d6', 'torchmd': '#eb6834'}
BOX_LINES = {'report': '-', 'images': (0, (4, 3))}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.frameon': False,
    'mathtext.default': 'regular',
    'pdf.fonttype': 42,
    'figure.dpi': 150,
})


def axes_style(ax):
    '''
    Grid and frame in the background, data in front
    '''
    ax.grid(alpha=0.3, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)
    ax.tick_params(colors=TEXT)


def box_set(npz_path):
    return Path(npz_path).stem.split('_', 1)[1]


def save(fig, outdir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=DPI)
    plt.close(fig)
    print(outdir / f'{name}.png')


def species_handles(composition, box, **kw):
    '''
    One legend entry per species, with the side of its box. box None: the boxes of the data
    '''
    sides, sign = (DATA_BOX, r'\approx') if box is None else (box, '=')
    return [plt.Line2D([], [], color=SPECIES[k][1], label=f'{SPECIES[k][0]} ($L {sign} {sides[k]:g}$ Å)', **kw)
            for k in SPECIES if k in composition]


def decades(ax, lo, hi, zero=None):
    '''
    At most ~7 decade ticks between lo and hi; zero is the height of the line labelled 0
    '''
    lo, hi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    step = max(1, int(np.ceil((hi - lo) / 6)))
    exps = list(range(hi, lo - 1, -step))
    ticks = [10.0 ** e for e in exps]
    labels = [f'$10^{{{e}}}$' for e in exps]
    if zero is not None:
        ticks, labels = ticks + [zero], labels + ['0']
    ax.set_yticks(ticks, labels)
    ax.set_yticks([], minor=True)


def comparison_label(name):
    if name.startswith('repeat'):
        return 'Repeated\nevaluation'
    if name.startswith('mixed vs isolated'):
        return f'Batched vs.\nisolated\n{name.split()[-1]}'
    if name.startswith('permuted'):
        return 'Permuted\nbatch order'
    if name.startswith('others'):
        return 'One system\nperturbed'
    if name.startswith('MUST FAIL'):
        return 'Negative\ncontrol'
    return name


def quantity(term):
    '''
    File prefix, name, y label and force term of a row term: energies end in _E, the rest are forces
    '''
    if term.endswith(ENERGY):
        return ('energy', 'Energy', r'Relative energy error  $|E - E_{ref}|\,/\,\max(|E_{ref}|,\,1)$',
                term[:-len(ENERGY)])
    return 'forces', 'Force', r'Relative force error  $\|F - F_{ref}\|\,/\,\|F_{ref}\|$', term


def plot_forces(npz_path, term, limit, outdir):
    '''
    Relative error of every system, grouped by comparison. Colour by species.
    L2 error of the forces, or error of the potential energy for the terms ending in _E
    '''
    prefix, name, ylabel, base = quantity(term)
    d = np.load(npz_path)
    composition = d['composition']
    rows = np.where(d['term'] == term)[0]
    rng = np.random.default_rng(0)

    # Exact zeros cannot go on a log axis: they are drawn one decade below the smallest error, labelled 0
    errors = d['errors'][rows]
    bottom = 10.0 ** np.floor(np.log10(np.nanmin(errors[errors > 0])))
    has_zero = bool((errors == 0).any())
    zero_line = bottom / 10

    fig, ax = plt.subplots(figsize=(WIDTH, 75 * MM), layout='constrained')
    present = [k for k in SPECIES if k in composition]
    width = 0.8 / len(present)
    for x, r in enumerate(rows):
        for j, key in enumerate(present):
            err = d['errors'][r, composition == key]
            # nan: the perturbed system itself, or species the control does not cover
            err = err[~np.isnan(err)]
            xs = x - 0.4 + (j + 0.5) * width + rng.uniform(-0.3, 0.3, err.size) * width
            _, color, marker = SPECIES[key]
            zero = err == 0
            ax.plot(xs[~zero], err[~zero], marker, color=color, ms=3, mew=0, alpha=0.75)
            ax.plot(xs[zero], np.full(zero.sum(), zero_line), marker, mfc='none', mec=color, ms=3, mew=0.6)

    ax.axhline(limit, color=TEXT, lw=0.8, ls='--')
    ax.text(len(rows) - 0.5, limit, 'tolerance ', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.axhline(EPS, color=TEXT, lw=0.6, ls=':')
    ax.text(len(rows) - 0.5, EPS, r'$\epsilon_{\mathrm{mach}}$ ', color=TEXT, ha='right', va='bottom', fontsize=7)

    top = max(limit, np.nanmax(errors))
    ax.set_yscale('log')
    ax.set_ylim(zero_line / 3 if has_zero else bottom / 3, top * 1000)
    decades(ax, bottom, top * 10, zero=zero_line if has_zero else None)

    ax.set_xticks(range(len(rows)), [comparison_label(str(d['comparison'][r])) for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{name} error, {TERMS[base]} ({BOXES[box_set(npz_path)]})')
    axes_style(ax)
    ax.grid(False, axis='x')
    ax.legend(handles=species_handles(composition, json.loads(str(d['box'])), ls='', ms=4), ncol=4, loc='upper left',
              handletextpad=0.2, columnspacing=1.2)
    for i, h in enumerate(ax.get_legend().legend_handles):
        h.set_marker(SPECIES[present[i]][2])

    save(fig, outdir, f'{prefix}_{box_set(npz_path)}_{base}')


def plot_forces_all(npz_paths, term, limit, outdir):
    '''
    One column per comparison, the three box sets side by side: worst system (marker) and median (dash)
    '''
    prefix, name, ylabel, base = quantity(term)
    fig, ax = plt.subplots(figsize=(WIDTH, 75 * MM), layout='constrained')
    labels = []
    # A median of exactly 0 (more than half of the energies agree to the last bit) cannot go on a log
    # axis: it is drawn one decade below the lowest tick, labelled 0
    zero_line, has_zero = 1e-18, False
    for j, path in enumerate(npz_paths):
        d = np.load(path)
        color, marker = BOX_STYLE[box_set(path)]
        # The must-fail row exists in one box set only
        rows = [r for r in np.where(d['term'] == term)[0] if not d['must_fail'][r]]
        labels = [comparison_label(str(d['comparison'][r])) for r in rows]
        for x, r in enumerate(rows):
            err = d['errors'][r]
            err = err[~np.isnan(err)]
            xs = x + (j - (len(npz_paths) - 1) / 2) * 0.22
            median = np.median(err)
            if median == 0:
                median, has_zero = zero_line, True
            ax.plot([xs, xs], [median, err.max()], color=color, lw=1.0)
            ax.plot(xs, median, '_', color=color, ms=7, mew=1.2)
            ax.plot(xs, err.max(), marker, color=color, ms=4.5, mew=0)

    ax.axhline(limit, color=TEXT, lw=0.8, ls='--')
    ax.text(len(labels) - 0.5, limit, 'tolerance ', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.axhline(EPS, color=TEXT, lw=0.6, ls=':')
    ax.text(len(labels) - 0.5, EPS, r'$\epsilon_{\mathrm{mach}}$ ', color=TEXT, ha='right', va='bottom', fontsize=7)

    ax.set_yscale('log')
    ax.set_ylim(zero_line / 3 if has_zero else 3e-18, limit * 1000)
    decades(ax, 1e-17, limit * 10, zero=zero_line if has_zero else None)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{name} error, {TERMS[base]}: worst system and median of each box set')
    axes_style(ax)
    ax.grid(False, axis='x')
    ax.legend(handles=[plt.Line2D([], [], color=BOX_STYLE[box_set(p)][0], marker=BOX_STYLE[box_set(p)][1], ls='',
                                  ms=4.5, mew=0, label=BOXES[box_set(p)]) for p in npz_paths],
              ncol=3, loc='upper left', handletextpad=0.2, columnspacing=1.2)
    save(fig, outdir, f'{prefix}_all_{base}')


def plot_nve_max(npz_paths, outdir):
    '''
    Worst system at each step: colour by reference engine, line style by box set
    '''
    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    s = slice(1, None)
    floor = 1e-17
    handles = []
    for path in npz_paths:
        d = np.load(path)
        threshold = json.loads(str(d['config']))['threshold']
        for engine, color in ENGINE_COLORS.items():
            ls = BOX_LINES[box_set(path)]
            ax.plot(d['steps'][s], np.maximum(d[f'div_{engine}'][s].max(axis=1), floor), color=color, lw=1.2, ls=ls)
            handles.append(plt.Line2D([], [], color=color, lw=1.2, ls=ls,
                                      label=f'vs. isolated {ENGINES[engine]}, {BOXES[box_set(path)]}'))

    ax.axhline(threshold, color=TEXT, lw=0.8, ls='--')
    ax.text(d['steps'][-1], threshold, 'tolerance', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(floor / 2, threshold * 30)
    decades(ax, floor, threshold * 10)
    ax.set_xlim(0, d['steps'][-1])
    ax.set_xlabel('Integration step')
    ax.set_ylabel('Max. atom displacement, worst system (Å)')
    ax.set_title('NVE divergence of the worst system')
    axes_style(ax)
    ax.legend(handles=handles, loc='lower right')
    save(fig, outdir, 'nve_max_all')


def plot_nve_species_time(npz_paths, outdir):
    '''
    Worst system of each species, box sets and engines pooled, against simulated time: the species
    do not share the time step, so the same step is not the same time
    '''
    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    floor = 1e-17
    data = [np.load(p) for p in npz_paths]
    threshold = json.loads(str(data[0]['config']))['threshold']
    t_max = 0.0
    for key, (label, color, _) in SPECIES.items():
        worst = np.max([d[f'div_{e}'][:, d['composition'] == key].max(axis=1) for d in data for e in ENGINES], axis=0)
        dt = float(data[0]['dt'][data[0]['composition'] == key][0])
        time = data[0]['steps'] * dt
        t_max = max(t_max, time[-1])
        ax.plot(time[1:], np.maximum(worst[1:], floor), color=color, lw=1.2, label=rf'{label} ($\Delta t$ = {dt:g} fs)')

    ax.axhline(threshold, color=TEXT, lw=0.8, ls='--')
    ax.text(t_max, threshold, 'tolerance', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(floor / 2, threshold * 30)
    decades(ax, floor, threshold * 10)
    ax.set_xlim(0, t_max)
    ax.set_xlabel('Simulated time (fs)')
    ax.set_ylabel('Max. atom displacement, worst system (Å)')
    ax.set_title('NVE divergence per species against simulated time')
    axes_style(ax)
    ax.legend(loc='lower right')
    save(fig, outdir, 'nve_species_time')


def plot_nve_final(npz_paths, outdir):
    '''
    Every system at the last step, grouped by species: colour by reference engine, filled for
    the first box set and hollow for the second
    '''
    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    rng = np.random.default_rng(0)
    handles = []
    ncols = len(npz_paths) * len(ENGINES)
    width = 0.8 / ncols
    for i, path in enumerate(npz_paths):
        d = np.load(path)
        threshold = json.loads(str(d['config']))['threshold']
        for j, (engine, color) in enumerate(ENGINE_COLORS.items()):
            kw = dict(color=color, ms=3.5, mew=0) if i == 0 else dict(mfc='white', mec=color, ms=3.5, mew=0.8)
            for x, key in enumerate(SPECIES):
                last = d[f'div_{engine}'][-1, d['composition'] == key]
                xs = x - 0.4 + (i * len(ENGINES) + j + 0.5) * width + rng.uniform(-0.3, 0.3, last.size) * width
                ax.plot(xs, last, 'o', alpha=0.8, **kw)
            handles.append(plt.Line2D([], [], marker='o', ls='', label=f'vs. isolated {ENGINES[engine]}, '
                                      f'{BOXES[box_set(path)]}', **kw))

    ax.axhline(threshold, color=TEXT, lw=0.8, ls='--')
    ax.text(len(SPECIES) - 0.5, threshold, 'tolerance ', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(1e-14, threshold * 300)
    decades(ax, 1e-14, threshold * 10)
    ax.set_xticks(range(len(SPECIES)), [label for label, _, _ in SPECIES.values()])
    ax.set_xlim(-0.5, len(SPECIES) - 0.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(f"Max. atom displacement at step {d['steps'][-1]} (Å)")
    ax.set_title('NVE divergence of every system at the last step')
    axes_style(ax)
    ax.grid(False, axis='x')
    ax.legend(handles=handles, ncol=2, loc='upper left')
    save(fig, outdir, 'nve_final')


def nve_top(npz_paths):
    '''
    Largest value drawn in any NVE figure, controls included: every figure shares the y axis
    '''
    top = 0.0
    for path in npz_paths:
        d = np.load(path)
        top = max(top, json.loads(str(d['config']))['threshold'],
                  *(np.nanmax(d[k]) for k in d.files if k.startswith('div_') or k.endswith('control_div')))
    return top


def plot_nve(npz_path, engine, top, outdir):
    '''
    Largest atom displacement between the batched and the isolated run, one line per system.
    top: largest value over all the figures, so that FiuMD and TorchMD share the y axis
    '''
    d = np.load(npz_path)
    config = json.loads(str(d['config']))
    steps, composition = d['steps'], d['composition']
    div = d[f'div_{engine}']
    # Step 0 is 0 by construction and log scale cannot show it
    s = slice(1, None)
    floor = 1e-17

    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    for i, key in enumerate(composition):
        ax.plot(steps[s], np.maximum(div[s, i], floor), color=SPECIES[key][1], lw=0.7, alpha=0.6)

    # Must-fail runs, only FiuMD: without them a flat line would prove nothing
    handles = species_handles(composition, config['box'], lw=1.2)
    if engine == 'fiumd':
        controls = [('control_div', r'wrong $\Delta t$', '-')]
        if 'image_control_div' in d.files:
            controls.append(('image_control_div', 'no periodic images', '--'))
        for name, label, ls in controls:
            ax.plot(steps[s], np.maximum(d[name][s], floor), color=TEXT, lw=1.1, ls=ls)
            handles.append(plt.Line2D([], [], color=TEXT, lw=1.1, ls=ls, label=f'control: {label}'))

    ax.axhline(config['threshold'], color=TEXT, lw=0.8, ls='--')
    ax.text(steps[-1], config['threshold'], 'tolerance', color=TEXT, ha='right', va='bottom', fontsize=7)

    ax.set_yscale('log')
    ax.set_ylim(floor / 2, top * 30)
    decades(ax, floor, top * 10)
    ax.set_xlim(0, steps[-1])
    ax.set_xlabel('Integration step')
    ax.set_ylabel('Max. atom displacement (Å)')
    ax.set_title(f'NVE divergence, batched vs. isolated {ENGINES[engine]} ({BOXES[box_set(npz_path)]})')
    axes_style(ax)
    # Between the tolerance and the controls, where no line passes
    ax.legend(handles=handles, ncol=2, loc='center right', bbox_to_anchor=(1.0, 0.68))

    save(fig, outdir, f'nve_{box_set(npz_path)}_{engine}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fuerzas', default='results/a2_nve/run5', help='folder with fuerzas_<box set>.npz')
    p.add_argument('--nve', default='results/a2_nve/run3/nve', help='folder with nve_<box set>.npz')
    p.add_argument('--outdir', default='results/a2_nve/figures')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = json.loads((Path(args.fuerzas) / 'config.json').read_text())['targets']
    force_paths = sorted(Path(args.fuerzas).glob('fuerzas_*.npz'))
    # Runs older than run5 have no energy rows
    terms = [t for t in [*TERMS, *(t + ENERGY for t in TERMS)] if t in targets]
    for path in force_paths:
        for term in terms:
            plot_forces(path, term, targets[term], outdir)
    for term in terms:
        plot_forces_all(force_paths, term, targets[term], outdir)

    nve_paths = sorted(Path(args.nve).glob('nve_*.npz'))
    top = nve_top(nve_paths)
    for path in nve_paths:
        for engine in ENGINES:
            plot_nve(path, engine, top, outdir)
    # Large boxes first: filled markers and continuous lines
    nve_paths = sorted(nve_paths, key=lambda p: list(BOX_LINES).index(box_set(p)))
    plot_nve_max(nve_paths, outdir)
    plot_nve_species_time(nve_paths, outdir)
    plot_nve_final(nve_paths, outdir)
