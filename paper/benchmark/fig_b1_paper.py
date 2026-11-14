'''
Paper figures of task B1 (one heterogeneous FiuMD batch vs the four proteins one after another),
one figure per file.

  b1_step_time    wall time per step against the ensemble size M
  b1_speedup      speedup of the heterogeneous batch over each sequential baseline
  b1_throughput   aggregate throughput, summed over the M systems
  b1_memory       peak GPU memory

Reads b1.csv and config.json written by bench_b1.py.
'''
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Same style as the A2 figures
from fig_a2_paper import MM, TEXT, WIDTH, axes_style, save

# Fixed order: label, colour, marker
MODES = {'het':         ('FiuMD, one heterogeneous batch', '#2a78d6', 'o'),
         'fiumd_seq':   ('FiuMD, four sequential batches', '#1baf7a', 's'),
         'torchmd_seq': ('TorchMD, four sequential runs',  '#eb6834', '^')}
HEIGHT = 75 * MM


def totals(csv_path):
    '''
    b1.csv -> M (nM,), {column: {mode: (nM,)}} of the rows of the whole workload
    '''
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r['species'] == 'total']
    M = np.array(sorted({int(r['M']) for r in rows}))
    cols = ['ms_median', 'ms_min', 'ms_max', 'peak_mem_mb']
    t = {c: {mode: np.full(M.size, np.nan) for mode in MODES} for c in cols}
    for r in rows:
        for c in cols:
            t[c][r['mode']][np.searchsorted(M, int(r['M']))] = float(r[c])
    return M, t


def new_axes(M, title):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), layout='constrained')
    ax.set_xscale('log', base=2)
    ax.set_xticks(M, [str(m) for m in M])
    ax.set_xticks([], minor=True)
    ax.set_xlabel('Ensemble size $M$ (systems, $M/4$ per protein)')
    ax.set_title(title)
    axes_style(ax)
    return fig, ax


def curve(ax, M, y, mode, **kw):
    label, color, marker = MODES[mode]
    ax.plot(M, y, marker=marker, color=color, lw=1.2, ms=4, label=label, **kw)


def plot_step_time(M, t, outdir):
    fig, ax = new_axes(M, 'Wall time per integration step')
    for mode in MODES:
        curve(ax, M, t['ms_median'][mode], mode)
        # Range over the repeats; narrower than the line almost everywhere
        ax.fill_between(M, t['ms_min'][mode], t['ms_max'][mode], color=MODES[mode][1], alpha=0.2, lw=0)
    ax.set_yscale('log')
    ax.set_ylabel('Time per step (ms)')
    ax.legend(loc='upper left')
    save(fig, outdir, 'b1_step_time')


def plot_speedup(M, t, outdir):
    med = t['ms_median']
    fig, ax = new_axes(M, 'Speedup of the heterogeneous batch')
    for mode in ['fiumd_seq', 'torchmd_seq']:
        speedup = med[mode] / med['het']
        curve(ax, M, speedup, mode)
        for m, s in zip(M, speedup):
            ax.annotate(f'{s:.1f}' if s < 10 else f'{s:.0f}', (m, s), xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=6.5, color=MODES[mode][1])
    ax.axhline(1, color=TEXT, lw=0.8, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(0.7, 150)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], ['1', '2', '5', '10', '20', '50', '100'])
    ax.set_yticks([], minor=True)
    ax.set_ylabel(r'Speedup  $t_{sequential}\,/\,t_{heterogeneous}$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f'over {l}' for l in labels], loc='upper left')
    save(fig, outdir, 'b1_speedup')


def plot_throughput(M, t, dt_fs, outdir):
    fig, ax = new_axes(M, 'Aggregate throughput')
    for mode in MODES:
        # ns/day summed over the M systems: M steps of dt every ms_median
        curve(ax, M, M * dt_fs * 1e-6 * 86400e3 / t['ms_median'][mode], mode)
    ax.set_yscale('log')
    ax.set_ylabel('Aggregate throughput (ns/day)')
    ax.legend(loc='upper left')
    save(fig, outdir, 'b1_throughput')


def plot_memory(M, t, outdir):
    fig, ax = new_axes(M, 'Peak GPU memory')
    for mode in MODES:
        curve(ax, M, t['peak_mem_mb'][mode] / 1024, mode)
    ax.set_yscale('log')
    ax.set_ylabel('Peak GPU memory (GiB)')
    ax.legend(loc='upper left')
    save(fig, outdir, 'b1_memory')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b1', default='results/b1', help='folder with b1.csv and config.json')
    p.add_argument('--outdir', default='results/b1/figures')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads((Path(args.b1) / 'config.json').read_text())
    M, t = totals(Path(args.b1) / 'b1.csv')

    plot_step_time(M, t, outdir)
    plot_speedup(M, t, outdir)
    plot_throughput(M, t, config['dt_fs'], outdir)
    plot_memory(M, t, outdir)
