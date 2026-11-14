'''
Benchmark fiumd 4 proteins one after the other 

Para M sistemas, con k = M/4 réplicas por especie:
  het          un ensemble FiuMD cln:k,trp:k,bbl:k,lambda:k
  fiumd_seq    4 ensembles FiuMD homogéneos de k réplicas (suma de los 4)
  torchmd_seq  4 motores TorchMD de k réplicas (suma de los 4)
'''
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heterogeneous_ensemble import (REPO_ROOT, PRECISIONS, parse_composition, parse_temps, load_molecules,
                                    make_initial_state, molecule_state, build_fiumd, build_torchmd, load_nnp,
                                    temperature_per_system)

SPECIES = ['cln', 'trp', 'bbl', 'lambda']
MODES = ['het', 'fiumd_seq', 'torchmd_seq']

# Las mismas condiciones en los dos motores
DT = 1.0      # fs
GAMMA = 1.0   # ps**-1; con 0, TorchMD se saltea el termostato
TEMP = 350.0


def time_full_steps(step_fn, nsteps, n_repeats, warmup, device):
    """Corre `nsteps` pasos consecutivos, repetido `n_repeats` veces, y
    devuelve tiempos por paso (ms) + memoria pico de GPU (MB)."""
    # Warmup: estabiliza kernels de GPU / caches
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    per_repeat_ms = []
    for _ in range(n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(nsteps):
            step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        per_repeat_ms.append((end - start) / nsteps * 1000.0)

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else float("nan")
    )
    return np.array(per_repeat_ms), peak_mem_mb



class OutOfMemory(Exception):
    pass

def run_engine(build_fn, device, *build_args, nsteps, n_repeats, warmup, **build_kwargs):
    """Arma un motor y lo mide, convirtiendo errores de memoria de GPU en
    OutOfMemory en vez de tirar abajo todo el barrido."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        step_fn = build_fn(*build_args, **build_kwargs)
        ms, mem = time_full_steps(step_fn, nsteps, n_repeats, warmup, device)
        return ms, mem
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM] {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise OutOfMemory(str(e)) from e
        raise

def het_composition(M):
    k = M // 4
    return parse_composition(','.join(f'{key}:{k}' for key in SPECIES))


def initial_state(commons, composition, seed):
    temps = parse_temps('', composition, TEMP)
    return make_initial_state(commons, composition, temps, seed)


def fiumd_engine(commons, state, model, device, dtype):
    n = len(state['composition'])
    # Un valor por sistema, como en heterogeneous_ensemble.py:847
    dt = torch.full((n,), DT, dtype=torch.float64)
    gamma = torch.full((n,), GAMMA, dtype=torch.float64)
    return build_fiumd(commons, state, gamma, dt, device, model, dtype)


def torchmd_engine(commons, state, key, model, device, dtype):
    # pos (natoms, 3, k), vel (k, natoms, 3)
    pos, vel = molecule_state(state, key)
    return build_torchmd(commons[key], pos, vel, TEMP, GAMMA, DT, device, model, dtype)


# run_engine espera una función que devuelva solo el step()
def fiumd_step(*args):
    return fiumd_engine(*args)['step']


def torchmd_step(*args):
    return torchmd_engine(*args)['step']


def check(commons, models, device, dtype, seed, nsteps=10):
    '''
    M = 4: unos pasos de cada modo, sin NaN y con una T razonable
    '''
    het = initial_state(commons, het_composition(4), seed)

    fiumd_states = {'het': het}
    fiumd_states.update({f'fiumd_seq {key}': species_state(het,key) for key in SPECIES})

    for name, state in fiumd_states.items():
        eng = fiumd_engine(commons, state, models['fiumd'], device, dtype)
        for _ in range(nsteps):
            eng['step']()
        ens = eng['ensemble']
        T = temperature_per_system(ens.masses.reshape(-1).detach().cpu(), ens.velocities.detach().cpu(),
                                   state['natoms_per_system'])
        finite = bool(torch.isfinite(ens.positions).all())
        print(f'{name:<20} finite = {finite}  T = {np.round(T, 1)}')
        del eng, ens

    for key in SPECIES:
        eng = torchmd_engine(commons, het, key, models['torchmd'], device, dtype)
        for _ in range(nsteps):
            _, _, T = eng['integrator'].step(niter=1)
        finite = bool(torch.isfinite(eng['system'].pos).all())
        print(f'{"torchmd_seq " + key:<20} finite = {finite}  T = {np.round(T, 1)}')
        del eng

def row(M, mode, species, ms, mem, k):
    return dict(M=M, mode=mode, species=species, ms_median=float(np.median(ms)), ms_min=float(ms.min()),
                ms_max=float(ms.max()), peak_mem_mb=mem, k=k)


def nan_row(M, mode, k):
    return dict(M=M, mode=mode, species='total', ms_median=np.nan, ms_min=np.nan, ms_max=np.nan,
                peak_mem_mb=np.nan, k=k)

def species_state(het, key):
    '''
    The systems of one species taken from het, so the three modes start from the same conformations
    '''
    pos, vel = molecule_state(het, key)   # (natoms, 3, k), (k, natoms, 3)
    k = pos.shape[-1]
    return dict(composition=[key] * k,
                natoms_per_system=np.full(k, pos.shape[0]),
                temps_per_system=np.full(k, TEMP),
                seed=het['seed'],
                pos_flat=torch.from_numpy(np.moveaxis(pos, -1, 0).reshape(-1, 3)),
                vel_flat=vel.reshape(-1, 3).float())

def measure(mode, M, commons, models, dev, device, dtype, seed, timing):
    '''
    Filas de un modo para un M. Puede levantar OutOfMemory
    '''
    k = M // 4
    het = initial_state(commons, het_composition(M), seed)

    if mode == 'het':
        ms, mem = run_engine(fiumd_step, dev, commons, het, models['fiumd'], device, dtype, **timing)
        return [row(M, mode, 'total', ms, mem, k)]

    rows = []
    for key in SPECIES:
        if mode == 'fiumd_seq':
            state = species_state(het, key)
            ms, mem = run_engine(fiumd_step, dev, commons, state, models['fiumd'], device, dtype, **timing)
        else:
            ms, mem = run_engine(torchmd_step, dev, commons, het, key, models['torchmd'], device, dtype, **timing)
        rows.append(row(M, mode, key, ms, mem, k))

    # En secuencia: los tiempos se suman y la memoria pico es la del más grande
    rows.append(dict(M=M, mode=mode, species='total',
                     ms_median=sum(r['ms_median'] for r in rows),
                     ms_min=sum(r['ms_min'] for r in rows),
                     ms_max=sum(r['ms_max'] for r in rows),
                     peak_mem_mb=max(r['peak_mem_mb'] for r in rows), k=k))
    return rows


def write_config(args, outdir):
    '''
    Todo lo necesario para repetir la corrida, escrito antes de empezar
    '''
    import subprocess
    import fiumd

    config = dict(vars(args))
    config.update(dt_fs=DT, gamma=GAMMA, temp=TEMP)
    config['fiumd_commit'] = subprocess.run(['git', '-C', str(Path(fiumd.__file__).parent), 'rev-parse', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip() or None
    config['torch_version'] = torch.__version__
    config['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    config['date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--M', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument('--nsteps', type=int, default=100)
    p.add_argument('--n_repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--nnp', choices=['cln', 'bbl', 'lambda'], default='lambda')
    p.add_argument('--precision', choices=list(PRECISIONS), default='float32')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datos_dir', default=str(REPO_ROOT / 'datos'))
    p.add_argument('--outdir', default=str(REPO_ROOT / 'results' / 'b1'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--check', action='store_true', help='M = 4, 10 pasos de cada modo, y sale')
    args = p.parse_args()

    if any(M % 4 for M in args.M):
        sys.exit(f'--M tiene que ser múltiplo de 4, got {args.M}')

    dev = torch.device(args.device)   # run_engine usa dev.type
    dtype = PRECISIONS[args.precision]
    commons = load_molecules(SPECIES, args.datos_dir)

    # Un modelo por motor: el ensemble convierte el suyo en el lugar
    models = dict(fiumd=load_nnp(args.nnp, commons, args.datos_dir, args.device, False, dtype),
                  torchmd=load_nnp(args.nnp, commons, args.datos_dir, args.device, True, dtype))

    if args.check:
        check(commons, models, args.device, dtype, args.seed)
        sys.exit()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(args, outdir)

    timing = dict(nsteps=args.nsteps, n_repeats=args.n_repeats, warmup=args.warmup)
    rows, dead = [], set()

    for i, M in enumerate(args.M):
        # Rota el orden para que la GPU caliente no favorezca siempre al mismo modo
        order = MODES[i % 3:] + MODES[:i % 3]
        print(f'M = {M}: {order}', flush=True)

        for mode in order:
            if mode in dead:
                rows.append(nan_row(M, mode, M // 4))
                continue
            try:
                new = measure(mode, M, commons, models, dev, args.device, dtype, args.seed, timing)
                rows += new
                print(f'  {mode:<12} {new[-1]["ms_median"]:8.2f} ms/paso', flush=True)
            except OutOfMemory:
                print(f'  {mode:<12} OOM, no se mide para M más grandes', flush=True)
                dead.add(mode)
                rows.append(nan_row(M, mode, M // 4))

            gc.collect()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

        # Después de cada M: torchmd_seq con M = 512 tarda
        pd.DataFrame(rows).to_csv(outdir / 'b1.csv', index=False)