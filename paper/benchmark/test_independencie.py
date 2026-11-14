'''
A2: the systems of a mixed FiuMD ensemble do not see each other.

Forces of every system in the mixed batch against the same system alone, in FiuMD
and in TorchMD (B=1), with priors only and with priors + NNP. Also a random
permutation of the systems and one system moved with the others fixed.
'''
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heterogeneous_ensemble import (REPO_ROOT, PRECISIONS, parse_composition, load_molecules,
                                        build_fiumd, build_torchmd, load_nnp, split_by_system)

# Box side per species in Å. 'data' keeps the box of each psf/xtc
BOXES = {
    'data': None,
    'report': {'cln': 30, 'trp': 40, 'bbl': 55, 'lambda': 70},
    'tight': {'cln': 25, 'trp': 26, 'bbl': 28, 'lambda': 30},
}

# Relative force error of the worst system, report targets
TARGETS = {'priors': 1e-14, 'nnp': 1e-6, 'priors_E': 1e-14, 'nnp_E':1e-6}

# The limit is max(target, NOISE_FACTOR * repeat): below the repeat noise nothing can be told apart
NOISE_FACTOR = 10

# Gaussian noise on every start frame, Å
POSITION_NOISE = 0.1

# Columns of fuerzas.csv, one row per comparison, term and species
COLUMNS = ['box_set', 'comparison', 'term', 'species', 'worst', 'system', 'moved', 'limit', 'floor', 'result']


def sample_positions(commons, composition, rng):
    '''
    One start frame per system, centred at the origin. All systems overlap,
    so cross-talk between them would show.
    There are more systems than frames, so every system gets its own noise: two equal
    systems could be swapped without the test seeing it
    '''
    pos = []
    for key in composition:
        coords = commons[key]['mol'].coords
        q = coords[:, :, rng.integers(coords.shape[2])].astype(np.float64)
        q = q + rng.normal(0.0, POSITION_NOISE, q.shape)
        pos.append(torch.from_numpy(q - q.mean(0)))
    return pos


def species_boxes(commons, box_set):
    if BOXES[box_set] is None:
        return None
    return {key: np.full(3, float(BOXES[box_set][key])) for key in commons}


def fiumd_forces(commons, composition, pos, boxes, nnp_model, device, dtype):
    '''
    One FiuMD ensemble with these systems. Returns the engine, the forces and potential energy per system
    '''
    natoms = np.array([commons[k]['mol'].numAtoms for k in composition])
    state = dict(composition=list(composition),
                 natoms_per_system=natoms,
                 temps_per_system=np.full(len(composition), 300.0),
                 pos_flat=torch.cat(pos),
                 vel_flat=torch.zeros(int(natoms.sum()), 3, dtype=torch.float64))

    eng = build_fiumd(commons, state, 0.0, 1.0, device, nnp_model, dtype, boxes)
    f, e = read_fiumd(eng, natoms)
    return eng, f, e


def read_fiumd(eng, natoms):

    ens = eng['ensemble']
    f = split_by_system(eng['ensemble'].forces.detach().cpu().double().clone(), 
    natoms)
    e = ens.energy.detach().reshape(-1).cpu().double().clone()
    
    return f, e 


def torchmd_forces(common, q, box, nnp_model, device, dtype):
    '''
    One system alone in TorchMD (B=1)
    '''
    vel = torch.zeros(1, q.shape[0], 3, dtype=torch.float64)
    eng = build_torchmd(common, q.numpy()[:, :, None], vel, 300.0, 0.0, 1.0, device, nnp_model, dtype, box)

    s = eng['system']
    e = eng['forces'].compute(s.pos,s.box,s.forces)

    return s.forces[0].detach().cpu().double().clone(), float(e[0])


def rel_err(f, ref):
    return float((f - ref).norm() / ref.norm())

def rel_err_e(e, ref):
    return float(abs(e -ref)/ max(abs(ref), 1.0))

def worst_per_species(errors, composition):
    '''
    errors: {system index: relative error} -> {species: (worst error, system index)}
    '''
    out = {}
    for i, e in errors.items():
        key = composition[i]
        if key not in out or e > out[key][0]:
            out[key] = (e, i)
    return out


def run_box_set(box_set, commons, composition, pos, models, rng, device, dtype):
    '''
    Rows of the table for one box set: (comparison, term, {system index: error}, must_fail)
    '''
    boxes = species_boxes(commons, box_set)
    natoms = np.array([commons[k]['mol'].numAtoms for k in composition])
    nsys = len(composition)
    rows = []

    perm = rng.permutation(nsys)
    moved = int(rng.integers(nsys))
    pos_moved = list(pos)
    # Random displacements: a new frame could be the same one, and a translation does not change the forces
    pos_moved[moved] = pos[moved] + torch.from_numpy(rng.normal(0.0, 0.1, tuple(pos[moved].shape)))

    for term in ('priors', 'nnp'):
        fiumd_model = models[term]['fiumd']
        torchmd_model = models[term]['torchmd']
        term_e = term + '_E'

        eng, mixed, mixed_e = fiumd_forces(commons, composition, pos, boxes, fiumd_model, device, dtype)

        # Same input again: the noise floor
        eng['ensemble'].set_positions(torch.cat(pos).to(device))
        repeat, repeat_e = read_fiumd(eng, natoms)
        rows.append(('repeat (noise floor)', term, {i: rel_err(repeat[i], mixed[i]) for i in range(nsys)}, False))
        rows.append(('repeat (noise floor)', term_e,
                     {i: rel_err_e(repeat_e[i], mixed_e[i]) for i in range(nsys)}, False))

        alone, alone_e = {}, {}
        for i, key in enumerate(composition):
            _, f, e = fiumd_forces(commons, [key], [pos[i]], boxes, fiumd_model, device, dtype)
            alone[i] = rel_err(mixed[i], f[0])
            alone_e[i] = rel_err_e(mixed_e[i], e[0])
        rows.append(('mixed vs isolated FiuMD', term, alone, False))
        rows.append(('mixed vs isolated FiuMD', term_e, alone_e, False))

        alone, alone_e = {}, {}
        for i, key in enumerate(composition):
            box = None if boxes is None else boxes[key]
            f, e = torchmd_forces(commons[key], pos[i], box, torchmd_model, device, dtype)
            alone[i] = rel_err(mixed[i], f)
            alone_e[i] = rel_err_e(mixed_e[i], e)
        rows.append(('mixed vs isolated TorchMD', term, alone, False))
        rows.append(('mixed vs isolated TorchMD', term_e, alone_e, False))

        # split_by_system recomputes the offsets for the permuted sizes
        comp_perm = [composition[j] for j in perm]
        _, f, e = fiumd_forces(commons, comp_perm, [pos[j] for j in perm], boxes, fiumd_model, device, dtype)
        rows.append(('permuted', term, {int(j): rel_err(f[n], mixed[j]) for n, j in enumerate(perm)}, False))
        rows.append(('permuted', term_e, {int(j): rel_err_e(e[n], mixed_e[j]) for n, j in enumerate(perm)}, False))

        eng['ensemble'].set_positions(torch.cat(pos_moved).to(device))
        f, e = read_fiumd(eng, natoms)
        if rel_err(f[moved], mixed[moved]) < 1e-3:
            sys.exit(f'system {moved} did not change when moved')
        if e[moved] == mixed_e[moved]:
            sys.exit(f'the energy of system {moved} did not change when moved')
        name = f'others with {moved} ({composition[moved]}) moved'
        rows.append((name, term, {i: rel_err(f[i], mixed[i]) for i in range(nsys) if i != moved}, False))
        rows.append((name, term_e, {i: rel_err_e(e[i], mixed_e[i]) for i in range(nsys) if i != moved}, False))

        # Must fail: the reference has another box, so the test has to see it
        if box_set == 'tight' and term == 'nnp':
            wrong, wrong_e = {}, {}
            for i, key in enumerate(composition):
                if key == 'lambda':
                    _, f, e = fiumd_forces(commons, [key], [pos[i]], None, fiumd_model, device, dtype)
                    wrong[i] = rel_err(mixed[i], f[0])
                    wrong_e[i] = rel_err_e(mixed_e[i], e[0])
            rows.append(('MUST FAIL: reference with data box', term, wrong, True))
            rows.append(('MUST FAIL: reference with data box', term_e, wrong_e, True))

    return rows, perm, moved


def print_table(box_set, rows, composition, moved):
    '''
    Prints the table and returns the number of failures and its rows for fuerzas.csv
    '''
    floor = {term: max(e for name, t, errs, _ in rows if t == term and name.startswith('repeat')
                       for e in errs.values()) for term in TARGETS}
    limit = {term: max(TARGETS[term], NOISE_FACTOR * floor[term]) for term in TARGETS}

    sides = 'psf/xtc' if BOXES[box_set] is None else ', '.join(f'{k} {v}' for k, v in BOXES[box_set].items())
    species = list(dict.fromkeys(composition))
    print(f'\nbox set {box_set} ({sides} Å)')
    print(f"{'comparison':<38} {'term':<9}" + ''.join(f'{k:>10}' for k in species) + f"{'limit':>10}  result")

    failed = 0
    records = []
    for name, term, errs, must_fail in rows:
        worst = worst_per_species(errs, composition)
        cells = ''.join(f'{worst[k][0]:>10.1e}' if k in worst else f"{'-':>10}" for k in species)
        top = max(errs.values())

        if name.startswith('repeat'):
            result = ''
        elif must_fail:
            ok = top > limit[term]
            result = 'FAIL as expected' if ok else 'PASSED: the test cannot see a wrong box'
            failed += not ok
        else:
            ok = top <= limit[term]
            result = 'pass' if ok else f'FAIL (system {max(errs, key=errs.get)})'
            failed += not ok

        print(f'{name:<38} {term:<9}{cells}{limit[term]:>10.1e}  {result}')

        # The name of the moved row carries the index, which changes between box sets
        base = 'others moved' if name.startswith('others with') else name
        for k in species:
            if k in worst:
                records.append(dict(box_set=box_set, comparison=base, term=term, species=k,
                                    worst=f'{worst[k][0]:.6e}', system=worst[k][1],
                                    moved=moved if base == 'others moved' else '',
                                    limit=f'{limit[term]:.6e}', floor=f'{floor[term]:.6e}',
                                    result=result or 'noise floor'))

    return failed, records


def write_config(args, outdir):
    '''
    Everything needed to repeat the run, written before it starts
    '''
    import fiumd

    config = dict(vars(args))
    config['targets'] = TARGETS
    config['noise_factor'] = NOISE_FACTOR
    config['position_noise'] = POSITION_NOISE
    config['box_sides'] = {b: BOXES[b] for b in args.boxes}
    config['fiumd_commit'] = subprocess.run(['git', '-C', str(Path(fiumd.__file__).parent), 'rev-parse', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip() or None
    config['torch_version'] = torch.__version__
    config['gpu'] = torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None
    config['date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


def write_csv(path, records):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(records)


if __name__ == '__main__':

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--composition', default='cln:32,trp:16,bbl:8,lambda:4')
    p.add_argument('--boxes', nargs='+', choices=list(BOXES), default=list(BOXES))
    p.add_argument('--nnp', choices=['cln', 'bbl', 'lambda'], default='lambda')
    p.add_argument('--precision', choices=list(PRECISIONS), default='float64')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datos_dir', default=str(REPO_ROOT / 'datos'))
    p.add_argument('--outdir', default=str(REPO_ROOT / 'results' / 'a2_nve'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    dtype = PRECISIONS[args.precision]
    composition = parse_composition(args.composition)
    commons = load_molecules(composition, args.datos_dir)

    # One model per derivative flag: FiuMD takes the energy model, TorchMD the one with forces
    models = dict(priors=dict(fiumd=None, torchmd=None),
                  nnp=dict(fiumd=load_nnp(args.nnp, commons, args.datos_dir, args.device, False, dtype),
                           torchmd=load_nnp(args.nnp, commons, args.datos_dir, args.device, True, dtype)))

    cutoff = models['nnp']['fiumd'].representation_model.cutoff_upper
    for box_set in args.boxes:
        for key, side in (BOXES[box_set] or {}).items():
            if key in commons and side < 2 * cutoff:
                sys.exit(f'{box_set}: {key} box {side} Å is below twice the NNP cutoff ({2 * cutoff} Å)')

    rng = np.random.default_rng(args.seed)
    pos = sample_positions(commons, composition, rng)
    species = list(dict.fromkeys(composition))

    print(f'{len(composition)} systems {dict((k, composition.count(k)) for k in species)}, '
          f'{args.precision}, {args.device}, NNP {args.nnp}, seed {args.seed}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(args, outdir)

    failed = 0
    records = []
    for box_set in args.boxes:
        rows, perm, moved = run_box_set(box_set, commons, composition, pos, models, rng, args.device, dtype)
        n, recs = print_table(box_set, rows, composition, moved)
        failed += n
        records += recs

        # Written after every box set, so a failure later keeps what is done
        write_csv(outdir / 'fuerzas.csv', records)
        errors = np.full((len(rows), len(composition)), np.nan)
        for r, (_, _, errs, _) in enumerate(rows):
            for i, e in errs.items():
                errors[r, i] = e
        np.savez(outdir / f'fuerzas_{box_set}.npz',
                 errors=errors,
                 comparison=np.array([r[0] for r in rows]),
                 term=np.array([r[1] for r in rows]),
                 must_fail=np.array([r[3] for r in rows]),
                 composition=np.array(composition),
                 perm=perm,
                 moved=moved,
                 box=json.dumps(BOXES[box_set]))

    print(f'\n{failed} failed' if failed else '\nall passed')
    sys.exit(1 if failed else 0)