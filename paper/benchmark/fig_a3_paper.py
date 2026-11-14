'''
Paper figures of task A3 (temperature ladder), FiuMD against TorchMD. One figure per file,
one panel per protein, target temperature on the x axis.

  a3_mean_T         <T> minus the target temperature (has to be zero)
  a3_sigma          std of T over the canonical value T sqrt(2/3N) (has to be one)
  a3_distribution   density of T / T_target against the canonical Gamma(3N/2), the four T pooled
  a3_epot           mean potential energy
  a3_rmsd           mean C-alpha RMSD to the experimental structure
  a3_angles         frames with a CA-CA-CA angle above 170 degrees

Also writes a3_ladder.csv, one row per (protein, target T, engine).

Reads the runs of heterogeneous_ensemble.py in results/sistemas_multiples:

  a3_escalera_fiumd_{1..4}     Latin square: every run holds the four temperatures at once,
                               one per protein
  a3_escalera_torchmd_T{T}     one homogeneous run per temperature

The engines are NOT paired: different number of replicas and different initial state. What is
compared are averages, each with its own error.

    python benchmark/fig_a3_paper.py
    python benchmark/fig_a3_paper.py --discard_ps 50 --plateau_ps 10 25 50 --outdir results/sistemas_multiples/figures
'''
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import gamma as gamma_dist, kstest

# Same style as the A2 and temperature-spike figures
from fig_a2_paper import MM, SPECIES, TEXT, WIDTH, axes_style, save
from fig_temp_spikes_paper import ENGINES, GAP_PS, QUANTILE, find_spikes
from fig_sistemas_multiples import REFERENCIAS, REPO_ROOT, leer_corrida, referencia_ca, rmsd

HEIGHT = 115 * MM
# Engines are shifted this much in x so that their error bars do not overlap
SHIFT_K = {'fiumd': -4.0, 'torchmd': 4.0}
# Same angle as results/temp_spikes/notes_causa_picos.md
ANGLE_DEG = 170.0
N_BOOT = 2000
# Error of <T>: plateau of the block curve over these block lengths (ps), each leaving at least
# MIN_BLOCKS blocks. BLOCK_CURVE_PS are the lengths drawn in a3_block_sem
PLATEAU_PS = (10.0, 25.0, 50.0)
MIN_BLOCKS = 10
BLOCK_CURVE_PS = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
# One frame every this much for the KS test, so that the samples are independent (tau_int of T
# is 0.4 to 0.9 ps in these runs)
KS_EVERY_PS = 5.0


def max_angle(pos):
    '''
    pos (frame, replica, atom, 3) -> largest CA-CA-CA angle of each replica (frame, replica), degrees.
    Close to 180 the force of the dihedral prior diverges (it divides by |r12 x r23|^2), which is
    what triggers the temperature spikes
    '''
    pos = pos.astype(np.float64)
    a = pos[:, :, :-2] - pos[:, :, 1:-1]
    b = pos[:, :, 2:] - pos[:, :, 1:-1]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max(-1)


def load(base, device, discard_ps):
    '''
    Folder with the a3_escalera_* runs -> {(protein, target T, engine): dict} with temp, epot and
    angle (frame, replica) and rmsd (replica, frame). The first `discard_ps` are dropped: Epot and
    RMSD start far from the equilibrium of each temperature
    '''
    groups = {}
    for run_dir in sorted(Path(base).glob('a3_escalera_*')):
        if not (run_dir / 'COMPLETED').exists():
            print(f'skipping {run_dir.name}: no COMPLETED file')
            continue
        cfg, trajs = leer_corrida(run_dir)
        for key, d in trajs.items():
            group = (key, float(d['temp_obj']), cfg['engine'])
            if group in groups:
                raise SystemExit(f'{group} is in two runs ({run_dir.name})')
            ref = referencia_ca(REPO_ROOT / 'datos' / 'experimental_structures' / REFERENCIAS[key])
            time_ps = d['steps'] * d['dt'] / 1000.0
            keep = time_ps > discard_ps
            if keep.sum() < 2:
                raise SystemExit(f'{group}: --discard_ps {discard_ps} leaves no frames')
            groups[group] = dict(
                natoms=d['pos'].shape[2], run=run_dir.name, time_ps=time_ps[keep],
                temp=d['temp'][keep].astype(np.float64), epot=d['epot'][keep].astype(np.float64),
                rmsd=rmsd(d['pos'], ref, device)[:, keep], angle=max_angle(d['pos'])[keep])
    if not groups:
        raise SystemExit(f'no complete a3_escalera_* runs in {base}')
    return groups


def block_sem(x, frame_ps, block_ps):
    '''
    Standard error of the mean of x (frame, replica) from blocks of `block_ps` inside each
    replica, the blocks of all replicas pooled. Also returns the number of blocks
    '''
    per_block = max(1, int(round(block_ps / frame_ps)))
    nblocks = x.shape[0] // per_block
    blocks = x[:nblocks * per_block].reshape(nblocks, per_block, -1).mean(1).ravel()
    return blocks.std(ddof=1) / np.sqrt(blocks.size), blocks.size


def block_mean(x, time_ps, plateau_ps):
    '''
    Mean of x (frame, replica) and its error at the plateau of the block curve (Flyvbjerg and
    Petersen): the largest block SEM over the block lengths in `plateau_ps` that still leave
    MIN_BLOCKS blocks. Valid when those blocks are much longer than the correlation time: true
    for T (tau_int 0.4 to 0.9 ps in these runs), not for Epot or RMSD
    '''
    frame_ps = time_ps[1] - time_ps[0]
    sems = [block_sem(x, frame_ps, b) for b in plateau_ps]
    sems = [s for s, n in sems if n >= MIN_BLOCKS]
    if not sems:
        raise SystemExit(f'no block length in {plateau_ps} ps leaves {MIN_BLOCKS} blocks')
    return x.mean(), max(sems)


def tau_int(x, frame_ps):
    '''
    Integrated autocorrelation time of x (frame, replica), the autocorrelation averaged over
    replicas and summed up to its first negative value
    '''
    d = x - x.mean(0)
    n = d.shape[0]
    f = np.fft.rfft(d, 2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(1)
    acf /= acf[0]
    cut = int(np.argmax(acf < 0)) if (acf < 0).any() else n
    return frame_ps * (0.5 + acf[1:cut].sum())


def replica_mean(x):
    '''
    Mean and standard error of x (replica, frame) with each replica as one independent sample.
    For what decorrelates slowly (Epot, RMSD, angles)
    '''
    m = x.mean(1)
    return m.mean(), m.std(ddof=1) / np.sqrt(m.size)


def analyse(groups, plateau_ps, rng):
    '''
    Adds to each group the (value, error) pairs that the figures plot
    '''
    for (key, t0, engine), g in groups.items():
        n = g['natoms']
        canonical = gamma_dist(a=1.5 * n, scale=2 * t0 / (3 * n))
        sigma = t0 * np.sqrt(2 / (3 * n))
        frame_ps = g['time_ps'][1] - g['time_ps'][0]
        temp = g['temp']

        # Replicas are resampled, not frames: the frames of one replica are correlated
        nrep = temp.shape[1]
        boot = [temp[:, rng.integers(0, nrep, nrep)].std() for _ in range(N_BOOT)]
        count, mask = find_spikes(temp, canonical.ppf(1 - QUANTILE), int(round(GAP_PS / frame_ps)))
        ks = kstest(temp[::max(1, int(round(KS_EVERY_PS / frame_ps)))].ravel(), canonical.cdf)

        mean, err = block_mean(temp, g['time_ps'], plateau_ps)
        # Block curve normalised by the frame-level SEM; blocks shorter than a frame are skipped
        curve = [(b, block_sem(temp, frame_ps, b)) for b in BLOCK_CURVE_PS if b >= frame_ps - 1e-9]
        g.update(mean_T=(mean - t0, err), tau_T=tau_int(temp, frame_ps),
                 block_curve=[(b, s / curve[0][1][0], n) for b, (s, n) in curve], sigma=(temp.std() / sigma, np.std(boot) / sigma),
                 sigma_no_spikes=temp[~mask].std() / sigma, spikes=int(count.sum()), max_T=temp.max(),
                 ks_p=ks.pvalue, epot=replica_mean(g['epot'].T), rmsd_mean=replica_mean(g['rmsd']),
                 angles=replica_mean(100.0 * (g['angle'] > ANGLE_DEG).T))
    return groups


def table(groups):
    '''
    One row per (protein, target T, engine). The FiuMD rows also carry the difference between
    engines over its combined error
    '''
    rows = []
    for key in SPECIES:
        for t0 in sorted({t for k, t, _ in groups if k == key}):
            for engine in ENGINES:
                g = groups.get((key, t0, engine))
                if g is None:
                    continue
                row = dict(protein=key, target_T=t0, engine=engine, run=g['run'], replicas=g['temp'].shape[1],
                           analysed_ps=g['time_ps'][-1] - g['time_ps'][0],
                           mean_T_minus_target=g['mean_T'][0], mean_T_err=g['mean_T'][1],
                           z_target=g['mean_T'][0] / g['mean_T'][1], tau_T_ps=g['tau_T'],
                           sigma_over_canonical=g['sigma'][0], sigma_err=g['sigma'][1],
                           sigma_no_spikes=g['sigma_no_spikes'], spikes=g['spikes'], max_T=g['max_T'],
                           ks_p=g['ks_p'], epot=g['epot'][0], epot_err=g['epot'][1],
                           rmsd=g['rmsd_mean'][0], rmsd_err=g['rmsd_mean'][1],
                           pct_frames_angle=g['angles'][0], pct_frames_angle_err=g['angles'][1])
                other = groups.get((key, t0, 'torchmd'))
                for name, field in (('z_engines_T', 'mean_T'), ('z_engines_epot', 'epot'),
                                    ('z_engines_rmsd', 'rmsd_mean'), ('z_engines_angle', 'angles')):
                    row[name] = ''
                    if engine == 'fiumd' and other is not None:
                        row[name] = (g[field][0] - other[field][0]) / np.hypot(g[field][1], other[field][1])
                rows.append(row)
    return rows


def panel_title(ax, groups, key, letter):
    temps = sorted({t for k, t, _ in groups if k == key})
    nrep = ' / '.join(str(groups[(key, temps[0], e)]['temp'].shape[1])
                      for e in ENGINES if (key, temps[0], e) in groups)
    ax.set_title(f'{SPECIES[key][0]}, {nrep} replicas')
    ax.set_title(letter, loc='left', fontweight='bold')
    return temps


def engine_handles(groups):
    return [Line2D([], [], color=color, marker=marker, ls='', ms=5, mew=0, label=label)
            for engine, (label, color, _, marker) in ENGINES.items() if any(e == engine for _, _, e in groups)]


def plot_against_T(groups, field, ylabel, outdir, name, reference=None, symmetric=False):
    '''
    `field` (value, error) against the target temperature, one panel per protein. `reference` is
    the horizontal line of what it has to give; without one, a thin line joins the points
    '''
    species = [k for k in SPECIES if any(kk == k for kk, _, _ in groups)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), layout='constrained')
    for ax, key, letter in zip(axes.flat, species, 'abcd'):
        temps = panel_title(ax, groups, key, letter)
        for engine, (label, color, ls, marker) in ENGINES.items():
            points = [(t, *groups[(key, t, engine)][field]) for t in temps if (key, t, engine) in groups]
            if not points:
                continue
            t, y, err = map(np.array, zip(*points))
            ax.errorbar(t + SHIFT_K[engine], y, yerr=err, fmt=marker, color=color, ms=5, mew=0,
                        elinewidth=1.0, capsize=2.5, zorder=3)
            if reference is None:
                ax.plot(t + SHIFT_K[engine], y, color=color, ls=ls, lw=0.8, zorder=2)
        if reference is not None:
            ax.axhline(reference, color=TEXT, lw=0.8, ls='--', zorder=1)
        if symmetric:
            top = max(abs(v) for v in ax.get_ylim())
            ax.set_ylim(-top, top)
        ax.set_xticks(temps)
        ax.set_xlim(temps[0] - 20, temps[-1] + 20)
        axes_style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('Target temperature (K)')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(handles=engine_handles(groups), ncol=2, loc='best')
    save(fig, outdir, name)


def plot_distribution(groups, outdir):
    '''
    T / T_target of the four temperatures pooled. Divided by its target, the instantaneous
    temperature of a protein follows the same Gamma at any temperature: if the thermostat gave a
    system the temperature of another one, the pooled histogram would be wider or have two peaks.
    The canonical curve is not a fit
    '''
    species = [k for k in SPECIES if any(kk == k for kk, _, _ in groups)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), layout='constrained')
    for ax, key, letter in zip(axes.flat, species, 'abcd'):
        panel_title(ax, groups, key, letter)
        n = next(g['natoms'] for (k, _, _), g in groups.items() if k == key)
        canonical = gamma_dist(a=1.5 * n, scale=2 / (3 * n))
        lo, hi = canonical.ppf([1e-4, 1 - 1e-4])
        bins = np.linspace(lo, hi, 45)
        for engine, (label, color, ls, _) in ENGINES.items():
            ratio = [(g['temp'] / t0).ravel() for (k, t0, e), g in groups.items() if k == key and e == engine]
            if ratio:
                ax.hist(np.concatenate(ratio), bins=bins, density=True, histtype='step', color=color, lw=1.2, ls=ls)
        grid = np.linspace(lo, hi, 400)
        ax.plot(grid, canonical.pdf(grid), color=TEXT, lw=1.0, ls=':')
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, None)
        axes_style(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r'$T\,/\,T_{target}$')
    for ax in axes[:, 0]:
        ax.set_ylabel('Probability density')
    axes.flat[0].legend(handles=[Line2D([], [], color=c, ls=ls, lw=1.2, label=l) for l, c, ls, _ in ENGINES.values()]
                        + [Line2D([], [], color=TEXT, lw=1.0, ls=':', label='canonical')], loc='upper right')
    save(fig, outdir, 'a3_distribution')


def plot_block_curve(groups, outdir):
    '''
    SEM of <T> against block length, over the SEM with one frame per block, one line per
    (target T, engine). The error of a3_mean_T is the largest value over PLATEAU_PS (shaded):
    a curve still rising there would mean blocks too short for the correlation time
    '''
    species = [k for k in SPECIES if any(kk == k for kk, _, _ in groups)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), layout='constrained', sharex=True)
    for ax, key, letter in zip(axes.flat, species, 'abcd'):
        temps = panel_title(ax, groups, key, letter)
        for engine, (label, color, ls, marker) in ENGINES.items():
            for t0 in temps:
                g = groups.get((key, t0, engine))
                if g is None:
                    continue
                b, r, n = map(np.array, zip(*g['block_curve']))
                ok = n >= MIN_BLOCKS
                ax.plot(b[ok], r[ok], color=color, ls=ls, lw=0.9, marker=marker, ms=3, mew=0, alpha=0.8)
        ax.axvspan(min(PLATEAU_PS), max(PLATEAU_PS), color=TEXT, alpha=0.08, lw=0, zorder=0)
        ax.axhline(1.0, color=TEXT, lw=0.8, ls=':', zorder=1)
        ax.set_xscale('log')
        ax.set_xticks(BLOCK_CURVE_PS, [f'{b:g}' for b in BLOCK_CURVE_PS])
        ax.set_ylim(0.5, None)
        axes_style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('Block length (ps)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'SEM$(\langle T\rangle)$ / SEM$_{\mathrm{frame}}$')
    axes.flat[0].legend(handles=engine_handles(groups), ncol=2, loc='lower right')
    save(fig, outdir, 'a3_block_sem')


if __name__ == '__main__':
    base = REPO_ROOT / 'results' / 'sistemas_multiples'
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default=str(base), help='folder with the a3_escalera_* runs')
    p.add_argument('--outdir', default=str(base / 'figures'))
    p.add_argument('--discard_ps', type=float, default=50.0, help='equilibration that is not analysed')
    p.add_argument('--plateau_ps', type=float, nargs='+', default=list(PLATEAU_PS),
                   help='block lengths whose largest SEM is the error of <T>')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = analyse(load(args.base, args.device, args.discard_ps), args.plateau_ps, np.random.default_rng(0))

    plot_against_T(groups, 'mean_T', r'$\langle T \rangle - T_{target}$ (K)', outdir, 'a3_mean_T',
                   reference=0.0, symmetric=True)
    plot_against_T(groups, 'sigma', r'$\sigma_T\,/\,\sigma_T^{canonical}$', outdir, 'a3_sigma', reference=1.0)
    plot_distribution(groups, outdir)
    plot_block_curve(groups, outdir)
    plot_against_T(groups, 'epot', 'Mean potential energy (kcal/mol)', outdir, 'a3_epot')
    plot_against_T(groups, 'rmsd_mean', r'Mean C$\alpha$ RMSD (Å)', outdir, 'a3_rmsd')
    plot_against_T(groups, 'angles', f'Frames with angle > {ANGLE_DEG:g}° (%)', outdir, 'a3_angles')

    rows = table(groups)
    with open(outdir / 'a3_ladder.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(outdir / 'a3_ladder.csv')

    print(f"\n{'protein':8s} {'T':>4s} {'engine':8s} {'rep':>4s} {'<T>-T':>7s} {'err':>5s} {'z':>6s} "
          f"{'sigma':>6s} {'no spk':>6s} {'spikes':>6s} {'KS p':>6s} {'Epot':>8s} {'RMSD':>6s} {'angle %':>8s}")
    for r in rows:
        print(f"{r['protein']:8s} {r['target_T']:4.0f} {r['engine']:8s} {r['replicas']:4d} "
              f"{r['mean_T_minus_target']:7.2f} {r['mean_T_err']:5.2f} {r['z_target']:6.2f} "
              f"{r['sigma_over_canonical']:6.3f} {r['sigma_no_spikes']:6.3f} {r['spikes']:6d} {r['ks_p']:6.3f} "
              f"{r['epot']:8.2f} {r['rmsd']:6.2f} {r['pct_frames_angle']:8.2f}")
