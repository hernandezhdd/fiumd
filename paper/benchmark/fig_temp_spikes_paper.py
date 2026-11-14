'''
Temperature spikes in FiuMD and TorchMD, from the long validation runs (same protein, same
settings, independent Langevin trajectories in each engine). One figure per file.

  temp_trace_<run>   hottest replica at each saved frame, both engines
  temp_ccdf_<run>    fraction of frames above T, against the canonical Gamma(3N/2, kT)
  temp_sigma         std of T over the canonical value, with and without the spikes
  temp_event_rate    spikes per replica and ns, with a bootstrap interval over replicas
  temp_trace_all     the traces of the four proteins in one figure
  temp_ccdf_all      the tails of the four proteins in one figure
  temp_summary       temp_sigma and temp_event_rate in one figure

Also writes temp_spikes.csv with the numbers behind the last two figures.

Reads <mol>_b<B>_obs.npy (engine, frame, replica, [epot, temp, rg]) and <mol>_b<B>_pasos.npy
written by benchmark_fiumd_vs_torchmd.py --validate_save_coords.
'''
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gamma as gamma_dist

# Same style as the A2 figures
from fig_a2_paper import MM, SPECIES, TEXT, WIDTH, axes_style, save

# Order of the first axis of obs.npy: label, colour, line style, marker
ENGINES = {'fiumd':   ('FiuMD',   '#2a78d6', '-',         'o'),
           'torchmd': ('TorchMD', '#eb6834', (0, (4, 3)), 's')}
NATOMS = {'cln': 10, 'trp': 20, 'bbl': 47, 'lambda': 80}

# A frame is hot above the 1 - 1e-7 quantile of the canonical distribution of T: with at most
# 2e6 frames per engine and run, chance alone gives 0.2 hot frames
QUANTILE = 1e-7
# Hot frames of one replica closer than this are the same spike: the thermostat (gamma 1/ps)
# takes a few ps to remove the energy, and T can cross the threshold more than once meanwhile
GAP_PS = 5.0
N_BOOT = 5000
# Runs of the combined figures, one per protein: the longest of each
PANEL = ['cln_b128', 'trp_b64', 'bbl_b16', 'lambda_b8']


def load(run_dir):
    '''
    Folder of one validation run -> dict with temp (engine, frame, replica) in K and its settings
    '''
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'run_metadata.json').read_text())
    key = meta['molecules'][0].split(':')[0]
    batch = meta['validate_batch_sizes'][0]
    prefix = run_dir / 'coords' / f'{key}_b{batch}'
    temp = np.array(np.load(f'{prefix}_obs.npy', mmap_mode='r')[..., 1])
    steps = np.load(f'{prefix}_pasos.npy')

    n, t0 = NATOMS[key], meta['temp']
    canonical = gamma_dist(a=1.5 * n, scale=2 * t0 / (3 * n))
    return dict(name=f'{key}_b{batch}', key=key, batch=batch, temp=temp, t0=t0, canonical=canonical,
                time_ns=steps * meta['dt'] * 1e-6, frame_ps=(steps[1] - steps[0]) * meta['dt'] * 1e-3,
                threshold=canonical.ppf(1 - QUANTILE), sigma=t0 * np.sqrt(2 / (3 * n)))


def find_spikes(temp, threshold, gap):
    '''
    temp (frame, replica) -> spikes per replica (replica,), mask (frame, replica) of the frames
    of a spike: from its first hot frame to `gap` frames after the last one
    '''
    hot = temp > threshold
    count = np.zeros(temp.shape[1], dtype=int)
    mask = np.zeros(temp.shape, dtype=bool)
    for r in range(temp.shape[1]):
        frames = np.flatnonzero(hot[:, r])
        if frames.size == 0:
            continue
        count[r] = 1 + (np.diff(frames) > gap).sum()
        for f in frames:
            mask[f:f + gap + 1, r] = True
    return count, mask


def analyse(run, rng):
    '''
    One row per engine: std of T with and without spikes, spike rate and its 95% interval.
    Replicas are resampled, not spikes: the spikes of one replica are not independent
    '''
    gap = int(round(GAP_PS / run['frame_ps']))
    ns_per_replica = run['temp'].shape[1] * run['frame_ps'] * 1e-3
    rows, counts = [], {}
    for j, engine in enumerate(ENGINES):
        temp = run['temp'][j]
        count, mask = find_spikes(temp, run['threshold'], gap)
        counts[engine] = count
        boot = count[rng.integers(0, count.size, (N_BOOT, count.size))].mean(axis=1) / ns_per_replica
        rows.append(dict(run=run['name'], engine=engine, replicas=count.size, ns_per_replica=ns_per_replica,
                         threshold_K=run['threshold'], mean_T=temp.mean(), max_T=temp.max(),
                         sigma_canonical=run['sigma'], sigma_all=temp.std(), sigma_no_spikes=temp[~mask].std(),
                         spikes=count.sum(), replicas_with_spikes=(count > 0).sum(),
                         rate=count.mean() / ns_per_replica,
                         rate_lo=np.percentile(boot, 2.5), rate_hi=np.percentile(boot, 97.5)))

    # Do both engines have the same rate? Shuffle the engine label of the replicas
    both = np.concatenate([counts['fiumd'], counts['torchmd']])
    n = counts['fiumd'].size
    observed = abs(counts['fiumd'].mean() - counts['torchmd'].mean())
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        perm = rng.permutation(both)
        diffs[i] = abs(perm[:n].mean() - perm[n:].mean())
    for row in rows:
        row['p_same_rate'] = (1 + (diffs >= observed).sum()) / (1 + N_BOOT)
    return rows


def run_label(run):
    return f"{SPECIES[run['key']][0]}, {run['batch']} replicas"


def engine_handles(**kw):
    return [Line2D([], [], color=color, ls=ls, label=label, **kw) for label, color, ls, _ in ENGINES.values()]


def plot_trace(run, outdir):
    '''
    Hottest replica at each saved frame. The trajectories are independent, so the spikes of the
    two engines fall at different times; what has to agree is how often and how high
    '''
    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    draw_trace(ax, run)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Temperature of the hottest replica (K)')
    ax.set_title(f'Temperature spikes, {run_label(run)}')
    ax.legend(handles=[Line2D([], [], color=c, lw=1.2, label=l) for l, c, _, _ in ENGINES.values()],
              ncol=2, loc='upper left')
    save(fig, outdir, f"temp_trace_{run['name']}")


def draw_trace(ax, run, top=None):
    for j, (label, color, ls, _) in enumerate(ENGINES.values()):
        ax.plot(run['time_ns'], run['temp'][j].max(axis=1), color=color, lw=0.5, alpha=0.8, rasterized=True)

    ax.axhline(run['threshold'], color=TEXT, lw=0.8, ls='--')
    ax.text(run['time_ns'][-1], run['threshold'], 'spike threshold ', color=TEXT, ha='right', va='bottom', fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(run['t0'], top or max(run['temp'].max() * 3, run['threshold'] * 10))
    ax.set_xlim(run['time_ns'][0], run['time_ns'][-1])
    axes_style(ax)


def plot_ccdf(run, outdir):
    '''
    Fraction of the frames above T, all replicas pooled. The canonical curve is not a fit
    '''
    fig, ax = plt.subplots(figsize=(WIDTH, 70 * MM), layout='constrained')
    draw_ccdf(ax, run)
    ax.set_xlabel('Temperature $T$ (K)')
    ax.set_ylabel('Fraction of frames above $T$')
    ax.set_title(f'Tail of the temperature distribution, {run_label(run)}')
    ax.legend(handles=engine_handles(lw=1.2) + [Line2D([], [], color=TEXT, lw=1.0, ls=':', label='canonical')],
              loc='upper right')
    save(fig, outdir, f"temp_ccdf_{run['name']}")


def draw_ccdf(ax, run, t_max=None, bottom=None):
    grid = np.geomspace(run['t0'] * 0.5, t_max or run['temp'].max() * 1.2, 600)
    nframes = run['temp'][0].size
    for j, (label, color, ls, _) in enumerate(ENGINES.values()):
        above = nframes - np.searchsorted(np.sort(run['temp'][j], axis=None), grid, side='right')
        ax.step(grid, np.where(above > 0, above / nframes, np.nan), where='post', color=color, lw=1.2, ls=ls)
    ax.plot(grid, run['canonical'].sf(grid), color=TEXT, lw=1.0, ls=':')

    ax.axvline(run['threshold'], color=TEXT, lw=0.8, ls='--')
    ax.text(run['threshold'], 0.5, ' spike threshold', color=TEXT, ha='left', va='top', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom or 0.3 / nframes, 2)
    axes_style(ax)


def panel(runs, draw, xlabel, ylabel, legend, outdir, name, **kw):
    '''
    One panel per protein, same axes in the four so that the proteins can be compared
    '''
    runs = [next(r for r in runs if r['name'] == n) for n in PANEL]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 115 * MM), layout='constrained', sharey=True)
    for ax, run, letter in zip(axes.flat, runs, 'abcd'):
        draw(ax, run, **kw)
        ax.set_title(run_label(run))
        ax.set_title(letter, loc='left', fontweight='bold')
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(handles=legend, loc='upper right', ncol=len(legend) if len(legend) < 3 else 1)
    save(fig, outdir, name)


def summary_axes(runs, title):
    fig, ax = plt.subplots(figsize=(WIDTH, 75 * MM), layout='constrained')
    summary_style(ax, runs)
    ax.set_title(title)
    return fig, ax


def summary_style(ax, runs):
    ax.set_xticks(range(len(runs)), [f"{SPECIES[r['key']][0]}\n{r['batch']} replicas × "
                                     f"{r['time_ns'][-1] - r['time_ns'][0]:.1f} ns" for r in runs])
    ax.set_xlim(-0.5, len(runs) - 0.5)
    axes_style(ax)
    ax.tick_params(axis='x', length=0)
    ax.grid(False, axis='x')


def plot_sigma(runs, rows, outdir):
    fig, ax = summary_axes(runs, 'Temperature fluctuations against the canonical value')
    draw_sigma(ax, runs, rows)
    save(fig, outdir, 'temp_sigma')


def draw_sigma(ax, runs, rows):
    for x, run in enumerate(runs):
        for j, (engine, (label, color, _, marker)) in enumerate(ENGINES.items()):
            row = next(r for r in rows if r['run'] == run['name'] and r['engine'] == engine)
            xs = x + (j - 0.5) * 0.3
            ax.plot(xs, row['sigma_all'] / row['sigma_canonical'], marker, color=color, ms=5, mew=0)
            ax.plot(xs, row['sigma_no_spikes'] / row['sigma_canonical'], marker, mfc='white', mec=color, ms=5, mew=1.0)
    ax.axhline(1, color=TEXT, lw=0.8, ls='--')
    ax.set_ylabel(r'$\sigma_T\,/\,\sigma_T^{canonical}$')
    ax.legend(handles=[Line2D([], [], color=c, marker=m, ls='', ms=5, mew=0, label=f'{l}, all frames')
                       for l, c, _, m in ENGINES.values()] +
                      [Line2D([], [], mec=c, mfc='white', marker=m, ls='', ms=5, mew=1.0, label=f'{l}, spikes removed')
                       for l, c, _, m in ENGINES.values()], ncol=2, loc='upper left')


def plot_event_rate(runs, rows, outdir):
    fig, ax = summary_axes(runs, 'Rate of temperature spikes')
    draw_event_rate(ax, runs, rows)
    save(fig, outdir, 'temp_event_rate')


def draw_event_rate(ax, runs, rows):
    for x, run in enumerate(runs):
        for j, (engine, (label, color, _, marker)) in enumerate(ENGINES.items()):
            row = next(r for r in rows if r['run'] == run['name'] and r['engine'] == engine)
            xs = x + (j - 0.5) * 0.3
            ax.errorbar(xs, row['rate'], yerr=[[row['rate'] - row['rate_lo']], [row['rate_hi'] - row['rate']]],
                        fmt=marker, color=color, ms=5, mew=0, elinewidth=1.0, capsize=2.5)
            ax.annotate(str(row['spikes']), (xs, row['rate_hi']), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=6.5, color=color)
    ax.set_yscale('log')
    # Room for the counts and the legend; an interval that reaches 0 leaves through the bottom
    ax.set_ylim(3e-3, 3 * max(r['rate_hi'] for r in rows))
    ax.set_ylabel('Spikes per replica and ns')
    ax.legend(handles=[Line2D([], [], color=c, marker=m, ls='', ms=5, mew=0, label=l)
                       for l, c, _, m in ENGINES.values()], ncol=2, loc='upper left')


def plot_summary(runs, rows, outdir):
    '''
    a) the excess of fluctuations and what is left without the spikes, b) how often they happen
    '''
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 125 * MM), layout='constrained', sharex=True)
    for ax, letter, draw in ((a, 'a', draw_sigma), (b, 'b', draw_event_rate)):
        summary_style(ax, runs)
        draw(ax, runs, rows)
        ax.set_title(letter, loc='left', fontweight='bold')
    save(fig, outdir, 'temp_summary')


if __name__ == '__main__':
    base = 'results_validacion_larga'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', default=[f'{base}/cln', f'{base}/trp', f'{base}/bbl', f'{base}/lambda_bs8'],
                   help='folders with run_metadata.json and coords/')
    p.add_argument('--outdir', default='results/temp_spikes')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    runs, rows = [], []
    for path in args.runs:
        run = load(path)
        runs.append(run)
        rows += analyse(run, rng)
        plot_trace(run, outdir)
        plot_ccdf(run, outdir)

    plot_sigma(runs, rows, outdir)
    plot_event_rate(runs, rows, outdir)
    plot_summary(runs, rows, outdir)

    if all(n in [r['name'] for r in runs] for n in PANEL):
        shown = [r for r in runs if r['name'] in PANEL]
        t_max = max(r['temp'].max() for r in shown)
        panel(runs, draw_trace, 'Time (ns)', 'Hottest replica (K)',
              [Line2D([], [], color=c, lw=1.2, label=l) for l, c, _, _ in ENGINES.values()],
              outdir, 'temp_trace_all', top=t_max * 6)
        panel(runs, draw_ccdf, 'Temperature $T$ (K)', 'Fraction of frames above $T$',
              engine_handles(lw=1.2) + [Line2D([], [], color=TEXT, lw=1.0, ls=':', label='canonical')],
              outdir, 'temp_ccdf_all', t_max=t_max * 1.2, bottom=0.3 / max(r['temp'][0].size for r in shown))

    with open(outdir / 'temp_spikes.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'run':12s} {'engine':8s} {'sigma/canonical':>16s} {'no spikes':>10s} {'spikes':>7s} "
          f"{'per replica-ns':>22s} {'max T':>8s} {'p same rate':>12s}")
    for r in rows:
        print(f"{r['run']:12s} {r['engine']:8s} {r['sigma_all'] / r['sigma_canonical']:16.3f} "
              f"{r['sigma_no_spikes'] / r['sigma_canonical']:10.3f} {r['spikes']:7d} "
              f"{r['rate']:8.3f} [{r['rate_lo']:.3f}, {r['rate_hi']:.3f}] {r['max_T']:8.0f} {r['p_same_rate']:12.3f}")
