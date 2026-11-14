import argparse
import json
import sys
import time
from pathlib import Path
import itertools


import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from heterogeneous_ensemble import (REPO_ROOT, PRECISIONS, parse_composition, parse_temps, load_molecules,
                                        make_initial_state, build_fiumd, build_torchmd, load_nnp, split_by_system)


# Box side per species in Å. 'data' keeps the box of each psf/xtc
BOXES = {
    'data': None,
    'report': {'cln': 30, 'trp': 40, 'bbl': 55, 'lambda': 70},
    'tight': {'cln': 25, 'trp': 26, 'bbl': 28, 'lambda': 30},
}


def species_boxes(commons, box_set):
    if BOXES[box_set] is None:
        return None
    return {key: np.full(3, float(BOXES[box_set][key])) for key in commons}


DT = {'cln': 1.0, 'trp': 1.0, 'bbl': 0.5, 'lambda': 0.5}

THRESHOLD = 1e-8


# Closest an atom may start to a periodic image of its own system, in Å
MIN_IMAGE_DISTANCE = 4.0


def image_distance(q, side):
    '''
    Smallest distance between an atom and a periodic image of any atom of the same system
    '''
    d = q[:, None, :] - q[None, :, :]
    shifts = [s for s in itertools.product((-1, 0, 1), repeat=3) if any(s)]
    return min(float((d + torch.tensor(s, dtype=q.dtype) * side).norm(dim=-1).min()) for s in shifts)


def check_start(commons, composition, state, box_set):
    '''
    A system that starts on top of its own image explodes, and the explosion
    amplifies rounding until mixed and isolated no longer agree
    '''
    boxes = species_boxes(commons, box_set)
    pos = split_by_system(state['pos_flat'], state['natoms_per_system'])
    bad = []
    for i, key in enumerate(composition):
        side = torch.as_tensor(commons[key]['mol'].box[:, 0] if boxes is None else boxes[key], dtype=torch.float64)
        d = image_distance(pos[i], side)
        if d < MIN_IMAGE_DISTANCE:
            bad.append(f'system {i} ({key}): {d:.2f} Å from its image in a {float(side.max()):.0f} Å box')
    if bad:
        sys.exit(f'box set {box_set}: systems start closer than {MIN_IMAGE_DISTANCE} Å to their own image\n  '
                 + '\n  '.join(bad))

def centred_state (commons, composition, temps, seed):
    '''
    Systems centred at the origin so they overlap
    '''

    state = make_initial_state(commons, composition, temps, seed)
    natoms = state['natoms_per_system']

    pos = [q.double() - q.double().mean(0) for q in split_by_system(state['pos_flat'], natoms)]
    state['pos_flat'] = torch.cat(pos)
    state['vel_flat'] = state['vel_flat'].double()

    return state

def run_fiumd(commons, state, dt, boxes, nnp_model, steps, device, dtype):
    '''
    Positions at every step, (steps + 1, natoms_total, 3)
    '''
    eng = build_fiumd(commons, state, 0.0, dt, device, nnp_model, dtype, boxes)
    ens = eng['ensemble']

    traj = torch.zeros(steps + 1, *ens.positions.shape, dtype=torch.float64)
    traj[0] = ens.positions.detach().cpu()
    for n in range(1, steps + 1):
        # Velocity Verlet only: the thermostat would add noise
        eng['integrator']()
        traj[n] = ens.positions.detach().cpu()
    return traj


def run_torchmd(common, q, v, temp, dt, box, nnp_model, steps, device, dtype):
    '''
    One system alone (B=1). Positions at every step, (steps + 1, natoms, 3)
    '''
    # gamma = 0: TorchMD still calls langevin, but with a zero coefficient it leaves the velocities unchanged
    eng = build_torchmd(common, q.numpy()[:, :, None], v.unsqueeze(0), temp, 0.0, dt, device, nnp_model, dtype, box)
    s = eng['system']

    traj = torch.zeros(steps + 1, *s.pos.shape[1:], dtype=torch.float64)
    traj[0] = s.pos[0].detach().cpu()
    for n in range(1, steps + 1):
        eng['integrator'].step(niter=1)
        traj[n] = s.pos[0].detach().cpu()
    return traj

def max_displacement(a, b):
    '''
    (steps + 1, natoms, 3) twice -> (steps + 1,): largest atom displacement per step
    '''
    return (a - b).norm(dim=-1).max(dim=-1).values.numpy()


def run_box_set(box_set, commons, composition, state, models, steps, device, dtype):
    boxes = species_boxes(commons, box_set)
    natoms = state['natoms_per_system']
    pos = split_by_system(state['pos_flat'], natoms)
    vel = split_by_system(state['vel_flat'], natoms)
    nsys = len(composition)

    # One dt per system, in the order of the composition
    dt_mixed = torch.tensor([DT[k] for k in composition], dtype=torch.float64)

    t0 = time.time()
    mixed = split_by_system(run_fiumd(commons, state, dt_mixed, boxes, models['fiumd'], steps, device, dtype)
                            .transpose(0, 1), natoms)
    mixed = [m.transpose(0, 1) for m in mixed]
    print(f'  mixed FiuMD: {time.time() - t0:.0f} s', flush=True)

    div = {'fiumd': np.zeros((steps + 1, nsys)), 'torchmd': np.zeros((steps + 1, nsys))}
    t0 = time.time()
    for i, key in enumerate(composition):
        alone = dict(composition=[key],
                     natoms_per_system=natoms[i:i + 1],
                     temps_per_system=state['temps_per_system'][i:i + 1],
                     pos_flat=pos[i],
                     vel_flat=vel[i])
        # Scalar dt: alone there is only one system
        traj = run_fiumd(commons, alone, DT[key], boxes, models['fiumd'], steps, device, dtype)
        div['fiumd'][:, i] = max_displacement(mixed[i], traj)

        box = None if boxes is None else boxes[key]
        traj = run_torchmd(commons[key], pos[i], vel[i], float(state['temps_per_system'][i]), DT[key], box,
                           models['torchmd'], steps, device, dtype)
        div['torchmd'][:, i] = max_displacement(mixed[i], traj)

        print(f'  isolated {i + 1}/{nsys} ({key}): {time.time() - t0:.0f} s', flush=True)

    # Must fail: lambda alone with the dt of cln. If it passes, the test cannot see a wrong dt
    i = composition.index('lambda') if 'lambda' in composition else nsys - 1
    key = composition[i]
    wrong_dt = 1.0 if DT[key] != 1.0 else 0.5
    alone = dict(composition=[key], natoms_per_system=natoms[i:i + 1],
                 temps_per_system=state['temps_per_system'][i:i + 1], pos_flat=pos[i], vel_flat=vel[i])
    traj = run_fiumd(commons, alone, wrong_dt, boxes, models['fiumd'], steps, device, dtype)
    control = dict(system=i, species=key, dt=wrong_dt, div=max_displacement(mixed[i], traj))

    return div, control

def print_table(box_set, div, control, composition, steps):
    species = list(dict.fromkeys(composition))
    print(f'\nbox set {box_set}: largest displacement mixed vs isolated at step {steps} (Å)')
    print(f"{'reference':<20}" + ''.join(f'{k:>10}' for k in species) + f"{'limit':>10}  result")

    failed = 0
    for engine, d in div.items():
        last = d[-1]
        cells = ''.join(f'{max(last[i] for i, c in enumerate(composition) if c == k):>10.1e}' for k in species)
        ok = last.max() <= THRESHOLD
        result = 'pass' if ok else f'FAIL (system {int(last.argmax())})'
        failed += not ok
        print(f'{"isolated " + engine:<20}{cells}{THRESHOLD:>10.1e}  {result}')

    last = control['div'][-1]
    ok = last > THRESHOLD
    failed += not ok
    print(f"MUST FAIL: system {control['system']} ({control['species']}) with dt {control['dt']} fs: "
          f"{last:.1e}  {'FAIL as expected' if ok else 'PASSED: the test cannot see a wrong dt'}")
    return failed

# The NNP sees an image closer than its cutoff; below the lower end the system starts in contact with it
IMAGE_WINDOW = (5.0, 11.0)


def image_frames(commons, counts, min_side, rng):
    '''
    Per species, the box side with the most frames whose closest image falls inside
    IMAGE_WINDOW, and counts[key] of those frames drawn without repetition
    '''
    sides, frames = {}, {}
    for key, k in counts.items():
        coords = torch.from_numpy(commons[key]['mol'].coords.astype(np.float64)).permute(2, 0, 1)
        best = []
        for side in range(int(np.ceil(min_side)), 80):
            d = np.array([image_distance(q, torch.full((3,), float(side), dtype=torch.float64)) for q in coords])
            ok = np.flatnonzero((d >= IMAGE_WINDOW[0]) & (d < IMAGE_WINDOW[1]))
            # Most frames first, then the smallest side
            best.append((len(ok), -side, ok))
        n, side, ok = max(best, key=lambda t: t[:2])
        if n < k:
            sys.exit(f'{key}: only {n} frames have their image within {IMAGE_WINDOW} Å, {k} requested')
        sides[key] = -side
        frames[key] = rng.choice(ok, size=k, replace=False)
    return sides, frames


def images_state(commons, composition, temps, seed, frames):
    '''
    Velocities as in centred_state, positions from the chosen frames
    '''
    state = centred_state(commons, composition, temps, seed)
    pos, used = [], dict.fromkeys(frames, 0)
    for key in composition:
        q = torch.from_numpy(commons[key]['mol'].coords[:, :, frames[key][used[key]]].astype(np.float64))
        used[key] += 1
        pos.append(q - q.mean(0))
    state['pos_flat'] = torch.cat(pos)
    return state


def image_control(commons, composition, state, model, steps, device, dtype):
    '''
    Must fail: the system closest to its image, alone, with and without its images.
    If both trajectories agree, the images play no role and the box set tests nothing
    '''
    boxes = species_boxes(commons, 'images')
    natoms = state['natoms_per_system']
    pos = split_by_system(state['pos_flat'], natoms)
    vel = split_by_system(state['vel_flat'], natoms)
    dist = [image_distance(q, torch.as_tensor(boxes[k], dtype=torch.float64)) for q, k in zip(pos, composition)]

    i = int(np.argmin(dist))
    key = composition[i]
    alone = dict(composition=[key], natoms_per_system=natoms[i:i + 1],
                 temps_per_system=state['temps_per_system'][i:i + 1], pos_flat=pos[i], vel_flat=vel[i])
    # Three times the side puts the closest image beyond the NNP cutoff
    far = {k: 3 * b for k, b in boxes.items()}
    with_images = run_fiumd(commons, alone, DT[key], boxes, model, steps, device, dtype)
    without = run_fiumd(commons, alone, DT[key], far, model, steps, device, dtype)

    # Closest image along the run: if it leaves the window, the system touched its image
    side = torch.as_tensor(boxes[key], dtype=torch.float64)
    along = [image_distance(with_images[n], side) for n in range(0, steps + 1, 50)]
    return dict(system=i, species=key, start=dist[i], closest=min(along), farthest=max(along),
                div=max_displacement(with_images, without))

def write_config(args, outdir):
    '''
    Everything needed to repeat the run, written before it starts
    '''
    import subprocess
    import fiumd

    config = dict(vars(args))
    config['dt_per_species'] = DT
    config['threshold'] = THRESHOLD
    config['box_sides'] = {b: BOXES[b] for b in args.boxes}
    config['fiumd_commit'] = subprocess.run(['git', '-C', str(Path(fiumd.__file__).parent), 'rev-parse', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip() or None
    config['torch_version'] = torch.__version__
    config['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    config['date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--composition', default='cln:32,trp:16,bbl:8,lambda:4')
    p.add_argument('--boxes', nargs='+', choices=list(BOXES) + ['images'], default=['report', 'images'])
    p.add_argument('--images_composition', default='cln:3,trp:16,bbl:8,lambda:4',
                   help='systems of the images box set; cln has only 3 frames that fit')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--temp', type=float, default=350)
    p.add_argument('--temps', default='', help='T by species, for example cln:300,trp:350')
    p.add_argument('--nnp', choices=['cln', 'bbl', 'lambda'], default='lambda')
    p.add_argument('--precision', choices=list(PRECISIONS), default='float64')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datos_dir', default=str(REPO_ROOT / 'datos'))
    p.add_argument('--outdir', default=str(REPO_ROOT / 'results' / 'a2_nve'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    dtype = PRECISIONS[args.precision]
    composition = parse_composition(args.composition)
    images_composition = parse_composition(args.images_composition)
    commons = load_molecules(composition + images_composition, args.datos_dir)

    # FiuMD takes the energy model, TorchMD the one with forces
    models = dict(fiumd=load_nnp(args.nnp, commons, args.datos_dir, args.device, False, dtype),
                  torchmd=load_nnp(args.nnp, commons, args.datos_dir, args.device, True, dtype))

    # Systems and start of every box set, all checked before the first step
    runs, frames = {}, None
    for box_set in args.boxes:
        if box_set == 'images':
            comp = images_composition
            counts = {k: comp.count(k) for k in dict.fromkeys(comp)}
            cutoff = models['fiumd'].representation_model.cutoff_upper
            BOXES['images'], frames = image_frames(commons, counts, 2 * cutoff, np.random.default_rng(args.seed))
            state = images_state(commons, comp, parse_temps(args.temps, comp, args.temp), args.seed, frames)
        else:
            comp = composition
            state = centred_state(commons, comp, parse_temps(args.temps, comp, args.temp), args.seed)
        check_start(commons, comp, state, box_set)
        runs[box_set] = (comp, state)

    print(f'{args.steps} steps, {args.precision}, {args.device}, NNP {args.nnp}, seed {args.seed}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(args, outdir)

    failed = 0
    for box_set in args.boxes:
        comp, state = runs[box_set]
        species = list(dict.fromkeys(comp))
        print(f'\nbox set {box_set} ({", ".join(f"{k} {BOXES[box_set][k]}" for k in species)} Å), '
              f'{len(comp)} systems {dict((k, comp.count(k)) for k in species)}', flush=True)
        div, control = run_box_set(box_set, commons, comp, state, models, args.steps, args.device, dtype)
        failed += print_table(box_set, div, control, comp, args.steps)

        extra = {}
        if box_set == 'images':
            ctl = image_control(commons, comp, state, models['fiumd'], args.steps, args.device, dtype)
            ok = ctl['div'][-1] > THRESHOLD
            failed += not ok
            print(f"MUST FAIL: system {ctl['system']} ({ctl['species']}) without its images: {ctl['div'][-1]:.1e}  "
                  f"{'FAIL as expected' if ok else 'PASSED: the images play no role'}")
            print(f"closest image of system {ctl['system']}: {ctl['start']:.1f} Å at the start, "
                  f"{ctl['closest']:.1f}-{ctl['farthest']:.1f} Å along the run")
            extra = dict(frames=json.dumps({k: v.tolist() for k, v in frames.items()}),
                         image_control_div=ctl['div'])

        np.savez(outdir / f'nve_{box_set}.npz',
                 steps=np.arange(args.steps + 1),
                 composition=np.array(comp),
                 dt=np.array([DT[k] for k in comp]),
                 div_fiumd=div['fiumd'],
                 div_torchmd=div['torchmd'],
                 control_div=control['div'],
                 config=json.dumps(dict(vars(args), threshold=THRESHOLD, box=BOXES[box_set])),
                 **extra)

    print(f'\n{failed} failed' if failed else '\nall passed')
    sys.exit(1 if failed else 0)