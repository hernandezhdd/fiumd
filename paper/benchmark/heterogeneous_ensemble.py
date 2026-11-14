'''
Testing ensamble for multi protein systems. 
    
'''
import argparse 
import sys 
import numpy as np 
from pathlib import Path
import torch 
import json 
from collections import Counter
import os
import signal 
#ROOTS AND PARAMETERS 

REPO_ROOT = Path(__file__).resolve().parent.parent

TIMEFACTOR = 48.88821  # fs -> unidades internas de tiempo
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR
BOLTZMAN = 0.001987191  # kcal/mol/K

MOLECULES = {"cln": "chignolin", "trp": "trpcage", "bbl": "bbl", "lambda": "lambda"}

FORCE_TERMS = {"BondForce", "RepulsionCGForce", "DihedralForce"}
FORCETERMS_PARAMS = ["Bonds", "RepulsionCG", "Dihedrals"]


# Atom type -> NNP embedding. Numbering of the cln, bbl and lambda checkpoints;
# the trp files use another one for K, L and R, so its checkpoint is not an option here
EMBEDDINGS = {"CAA": 1, "CAG": 2, "CAF": 3, "CAY": 4, "CAD": 5, "CAE": 6, "CAW": 7, "CAP": 8, "CAN": 9, "CAQ": 10,
              "CAH": 11, "CAS": 12, "CAT": 13, "CAV": 14, "CAM": 15, "CAR": 19, "CAK": 20, "CAL": 21, "CAI": 22}

RESUME_CHECK = ['engine', 'composition', 'temp', 'temps', 'dt', 'dts', 'gamma', 'gammas', 'nnp', 'steps', 'stride', 'seed', 'device', 'initial_state', 'precision']

PRECISIONS = {'float32': torch.float32, 'float64': torch.float64}



def parse_composition(text):
    """
    'cln:2,trp' -> ['cln', 'cln', 'trp']
    """
    comp = []

    for item in text.strip().split(","):
        key, _, n = item.strip().partition(":")

        comp.extend([key] * int(n or 1))
    return comp

def parse_temps (text, composition, default, flag='--temps'):
    '''
    cln:300, trp:330 -> {species: value}, default for the species not given.
    Also used for --dts and --gammas
    '''

    temps = {k: default for k in dict.fromkeys(composition)}
    
    for item in text.strip().split(","):
        item = item.strip()
        if not item:
            continue 

        key, _, t = item.partition(":")
        if key not in temps:
            sys.exit(f'{flag}: {key} is not in composition {list(temps)}')
        temps[key] = float (t)    

    return temps 

def load_common_data(molecule_name, molecule_long_name, datos_dir):

    from moleculekit.molecule import Molecule 

    forcefield = f"{datos_dir}/{molecule_name}/{molecule_name}_prior_force_field.yaml"
    topology = f"{datos_dir}/{molecule_name}/{molecule_name}.psf"
    starting_xtc = f"{datos_dir}/{molecule_name}/{molecule_name}_kcenters_32clusters_coords.xtc"
    coord_file = f"{datos_dir}/{molecule_name}/{molecule_long_name}_5120_ca_coords.npy"
    embeddings_path = f"{datos_dir}/{molecule_name}/{molecule_long_name}_ca_embeddings.npy"
    model_path = f"{datos_dir}/{molecule_name}/model_modified.ckpt"

    mol = Molecule(topology)
    mol.read(starting_xtc)

    full_coords = np.load(coord_file)
    embeddings_raw = np.load(embeddings_path).astype(int)

    return dict(
        mol=mol,
        forcefield=forcefield,
        full_coords=full_coords,
        embeddings_raw=embeddings_raw,
        model_path=model_path,
    )


def load_molecules(composition, datos_dir):
    """
    Loads each molecule and checks the molecular mass
    
    """
    commons = {}

    for key in dict.fromkeys(composition):

        if key not in MOLECULES:
            sys.exit(f"no se de donde cargar '{key}'; las conocidas son {list(MOLECULES)}")

        commons[key] = load_common_data(key, MOLECULES[key], datos_dir)

        masas = np.asarray(commons[key]["mol"].masses)
        if not (masas > 0).all():
            sys.exit(f"{key}: hay masas <= 0 en el psf ({np.unique(masas)}). "
                     f"Hay que llenarlas antes de correr")
    return commons


def split_by_system(flat, natoms_per_system):
    """
    (total_atoms, d) >> list of (natoms_i, d) one per system
    """

    out, o = [], 0

    for n in natoms_per_system:

        out.append(flat[o:o + n])
        o +=n
    
    return out

def temperature_per_system(mass_flat, vel_flat, natoms_per_system):
    '''
    Temperature in each system
    '''
    ekin_atom = 0.5 * (mass_flat.unsqueeze(1) * vel_flat **2 ).sum(-1)
    ekin = np.array([float(s.sum()) for s in split_by_system(ekin_atom.unsqueeze(1), natoms_per_system)])

    T = 2.0 *ekin / (3.0 *natoms_per_system *BOLTZMAN)

    return T

def sample_initial_positions (mol, batch_size, rng):

    nframes = mol.coords.shape[2]
    frame_idx = rng.choice(nframes, size=batch_size, replace=batch_size > nframes)
    return np.ascontiguousarray(mol.coords[:, :, frame_idx])


def sample_initial_velocities(masses, temp, batch_size):
    '''
    Returns (batch, natoms, 3) 
    '''
    import torchmd.integrator as integrator 
       
    return integrator.maxwell_boltzmann(masses, T=temp, replicas = batch_size)

def check_bonds(init_positions, mol, bond_range= (2.5, 5.5)):

    pos = torch.as_tensor (np.asarray(init_positions), dtype = torch.float32)
    pos = pos.permute(2,0,1) # permute to obtain (batch, natoms, 3)

    if len (mol.bonds) == 0:
        return # no bonds to check

    b = torch.as_tensor (np.asarray(mol.bonds), dtype = torch.long)
    d = (pos[:, b[:,0]] - pos[:, b[:,1]]).norm(dim = -1)

    mean = float (d.mean())

    lo, hi = bond_range

    if not np.isfinite(mean) or not (lo <= mean <= hi):
        raise ValueError (f'Bond distance is no in the expected range, mean value: {mean}, range: {bond_range}')

    return 

def make_initial_state (commons, composition, temps, seed):
    '''
    Positions and velocities for molecule, identical for multiprotein systems and reference. 
    
    '''
    count = Counter (composition)
    pos_per_mol , vel_per_mol = {}, {}

    rng = np.random.default_rng(seed)
    torch_state = torch.random.get_rng_state()

    try:
        torch.manual_seed (seed)
        
        for key, k in count.items():

            mol = commons[key]['mol']

            pos = sample_initial_positions(mol, k, rng)
            check_bonds(pos, mol)

            mass = torch.tensor(mol.masses).unsqueeze(1).float()

            vel = sample_initial_velocities(mass, temps[key], k)

            pos_per_mol[key] = pos
            vel_per_mol[key] = vel

    finally:
        torch.random.set_rng_state(torch_state)

    counted = Counter ()
    pos_flat, vel_flat = [], []

    for key in composition:
        j = counted[key]
        counted[key] +=1

        pos_flat.append(torch.tensor(pos_per_mol[key][:,:,j], dtype = torch.float32))

        vel_flat.append(vel_per_mol[key][j].float())
    
    return dict(
        composition=list(composition),
        natoms_per_system=np.array([commons[k]['mol'].numAtoms for k in composition]),
        temps_per_system=np.array([temps[k] for k in composition], dtype=float),
        seed=int(seed),
        pos_flat=torch.cat(pos_flat, dim=0),
        vel_flat=torch.cat(vel_flat, dim=0),
        )


def check_initial_temperatures(vel_flat, commons, composition, temps, max_z=5.0):
    '''
    T0 per system against its target. Expected spread with 3N degrees of
    freedom: sigma = T * sqrt(2 / 3N). Exits if a species mean is off by more
    than max_z standard errors.
    '''
    natoms = np.array([commons[k]['mol'].numAtoms for k in composition])
    mass_flat = torch.cat([torch.as_tensor(np.asarray(commons[k]['mol'].masses), dtype=vel_flat.dtype)
                           for k in composition])
    t_target = np.array([temps[k] for k in composition])

    t0 = temperature_per_system(mass_flat, vel_flat, natoms)
    sigma = t_target * np.sqrt(2.0 / (3.0 * natoms))

    if not np.isfinite(t0).all():
        sys.exit(f'T0 has NaN or inf: {t0}')

    if len(composition) <= 16:
        print(f"{'system':>6} {'mol':<7} {'N':>4} {'T target':>9} {'T0':>8} {'sigma':>7}")
        for i, key in enumerate(composition):
            print(f"{i:>6} {key:<7} {natoms[i]:>4} {t_target[i]:>9.1f} {t0[i]:>8.1f} {sigma[i]:>7.1f}")

    print(f"{'mol':<7} {'k':>4} {'T target':>9} {'<T0>':>8} {'std T0':>7} {'sigma':>7} {'z':>6}")
    comp = np.array(composition)
    for key, k in Counter(composition).items():
        m = comp == key
        mean = t0[m].mean()
        std = t0[m].std(ddof=1) if k > 1 else float('nan')
        z = (mean - temps[key]) / (sigma[m][0] / np.sqrt(k))
        print(f"{key:<7} {k:>4} {temps[key]:>9.1f} {mean:>8.1f} {std:>7.1f} {sigma[m][0]:>7.1f} {z:>6.2f}")
        if abs(z) > max_z:
            sys.exit(f'{key}: <T0> = {mean:.1f} K is {z:.1f} standard errors away from {temps[key]} K')

    return t0

def save_state(path, state):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    save_state= np.savez(path,
                        composition=np.array(state['composition']),
                        natoms_per_system = state['natoms_per_system'],
                        temps_per_system = state['temps_per_system'],
                        seed = state['seed'],
                        pos_flat = state['pos_flat'].numpy(),
                        vel_flat = state['vel_flat'].numpy(),   
                        )
    return save_state

def molecule_state(state, key):
    
    segs_pos = split_by_system(state['pos_flat'], state['natoms_per_system'])
    segs_vel = split_by_system(state['vel_flat'], state['natoms_per_system'])

    idx = [i for i, k in enumerate(state['composition']) if k == key]
    pos = np.stack([segs_pos[i] for i in idx], axis = -1)
    vel = torch.as_tensor (np.stack([segs_vel[i] for i in idx], axis = 0))

    return np.ascontiguousarray(pos), vel


def load_state(path, composition, temps, commons):
    '''
    Reads npz and verifies that is from this run 
    '''

    if not Path(path).exists():
        sys.exit(f'{path} does not exist')

    z = np.load(path)
    saved = [str(k) for k in z['composition']]
    if saved != list(composition):
        sys.exit(f'{path} is from composition {saved}')

    temps_per_system = np.array([temps[k] for k in composition], dtype=float)
    if not np.array_equal(z['temps_per_system'], temps_per_system):
        sys.exit(f"{path} is at T = {z['temps_per_system'].tolist()} K, expected {temps_per_system.tolist()}")

    natoms = np.array ([commons[k]['mol'].numAtoms for k in composition])
    if not np.array_equal(z['natoms_per_system'], natoms):
        sys.exit(f"{path}: natoms per system {z['natoms_per_system'].tolist()}, expected {natoms.tolist()}")

    return dict(composition = saved,
                natoms_per_system = natoms,
                temps_per_system = temps_per_system,
                seed = int(z['seed']),
                pos_flat = torch.from_numpy(z['pos_flat']),
                vel_flat = torch.from_numpy(z['vel_flat']),)

def make_run_dir(outdir, run_id):
    
    run_dir = Path (outdir) / run_id
    run_dir.mkdir(parents = True, exist_ok =True)

    if any (run_dir.iterdir()):
        sys.exit(f'{run_id} already in use, choose another id')
    
    return run_dir

def build_fiumd (commons, state,gamma,dt, device, nnp_model, dtype, boxes = None):
    '''
    Mixed Fiumd ensemble, priors only
    '''

    import fiumd
    import fiumd.ensembles
    import fiumd.forcefields.forcefield
    import fiumd.integrator
    import fiumd.parameters
    import fiumd.systems

    composition = state['composition']

    per_species = {}

    for key in dict.fromkeys(composition):
        mol = commons[key]['mol']

        box = mol.box[:, 0] if boxes is None else np.asarray(boxes[key], dtype = float)
        if not (box > 0).all():
            sys.exit(f'{key}: box must be >0, got {box}')

        ff = fiumd.forcefields.forcefield.ForceField.create(mol, commons[key]['forcefield'])

        params = fiumd.parameters.Parameters(ff, mol, FORCETERMS_PARAMS, precision = dtype, device = device )


        per_species[key] = dict(mol = mol, 
                                params = params, 
                                box = torch.tensor(box, device=device, dtype = dtype))

    #Per system 
    systems = []
    for key in composition:
        s = per_species[key]

        systems.append(fiumd.systems.System(molecule = s['mol'], 
                                            force_terms = FORCE_TERMS, 
                                            params = s['params'],
                                            box = s['box']))

    z = None 

    if nnp_model is not None: 
        z = torch.cat([nnp_embeddings(commons[k]['mol']) for k in composition]).to(device)

    ensemble = fiumd.ensembles.Ensemble(systems, 
                                       nnp_model = nnp_model, 
                                       nnp_embeddings = z,
                                       device = device, 
                                       dtype= dtype)

    ensemble.to(device)
    ensemble.eval()

    ensemble.set_positions(state['pos_flat'].clone().to(device))
    ensemble.set_velocities(state['vel_flat'].clone().to(device))

    temperature = torch.tensor(state['temps_per_system'], dtype = dtype, device = device)

    vv = fiumd.integrator.VelocityVerletIntegrator(ensemble = ensemble, dt = dt)
    lang = fiumd.integrator.LangevinThermostat (gamma=gamma, temperature=temperature, dt = dt, ensemble=ensemble)

    vv.to(device).eval()
    lang.to(device).eval()

    def step():
        vv()
        lang()

    return dict(ensemble=ensemble, 
                integrator = vv, 
                thermostat = lang, 
                step = step)


def build_torchmd (common, pos, vel, temp,gamma, dt, device, nnp_model, dtype, box=None):
    '''
    TorchMD engine for one species. 
    
    '''
    from torchmdnet import calculators
    import torchmd.forcefields.forcefield
    import torchmd.forces
    import torchmd.integrator
    import torchmd.parameters
    import torchmd.systems

    mol = common['mol']
    nreplicas = pos.shape[2]

    ff = torchmd.forcefields.forcefield.ForceField.create(mol, common['forcefield'])
    parameters = torchmd.parameters.Parameters(ff, mol, precision = dtype, device = device)

     # External expects (nreplicas, natoms)
    
    external = None 

    if nnp_model is not None:
        emb = nnp_embeddings(mol).unsqueeze(0).repeat(nreplicas, 1).to(device)
        external = calculators.External(nnp_model, emb, device = device, dtype = dtype)

    # cutoff = None: RepulsionCGForce in FiuMD has no cutoff
    forces = torchmd.forces.Forces(parameters,
                                   terms = FORCETERMS_PARAMS,
                                   external = external,
                                   cutoff = None,
                                   rfa = False,
                                   switch_dist = 7.5,
                                   exclusions = ("bonds",))

    system = torchmd.systems.System(mol.numAtoms, nreplicas = nreplicas, precision = dtype, device = device)
    system.set_positions(pos)
    system.set_box(mol.box[:, :1] if box is None else np.asarray(box, dtype = float).reshape(3,1)) # one box, same as FiuMD
    system.set_velocities(vel)

    # TorchMD starts with zero forces; FiuMD already has them after set_positions
    forces.compute(system.pos, system.box, system.forces)

    integrator = torchmd.integrator.Integrator(system, forces, dt, device, gamma = gamma, T = temp)

    def step():
        integrator.step(niter = 1)

    return dict(system = system,
                forces = forces,
                integrator = integrator,
                step = step)

def stop_if_nan (n, names, epot, temp, pos):

    for i, name in enumerate(names):
        if not (np.isfinite(epot[i]) and np.isfinite(temp[i]) and np.isfinite(pos[i]).all()):
            sys.exit(f'Nan at step{n}, system {name}: epot {epot[i]}, T = {temp[i]}')

def nnp_embeddings (mol):
    return torch.tensor([EMBEDDINGS[t] for t in mol.atomtype], dtype = torch.long)

def load_nnp(nnp_key, commons, datos_dir, device, derivative, dtype):

    from torchmdnet.models.model import load_model

    natoms_max = max (c['mol'].numAtoms for c in commons.values())
    path = f'{datos_dir}/{nnp_key}/model_modified.ckpt'
    model = load_model(path, derivative=derivative, max_num_neighbors = torch.tensor(natoms_max, device = device))

    return model.to(device, dtype = dtype )

def run_fiumd (eng, state, steps, stride, run_dir, save_every, device):
    ens = eng['ensemble']
    composition = state['composition']
    natoms = state['natoms_per_system']
    names = [f'{i} ({key})' for i, key in enumerate(composition)]
    mass_flat = ens.masses.reshape(-1).detach().cpu()

    size = min(save_every, steps // stride)
    progress = max(steps // 20, 1)

    saved_steps = np.zeros(size, dtype=int)

    dtype = ens.positions.detach().cpu().numpy().dtype

    epot = np.zeros((size, len(composition)), dtype=dtype)
    temp = np.zeros((size, len(composition)), dtype=dtype)
    pos = np.zeros((size, int(natoms.sum()), 3), dtype=dtype)

    start, part = resume_point(run_dir, 'checkpoint.npz', dict.fromkeys(composition), device)
    frame = 0
    try: 
        for n in range(start + 1, steps + 1):        
            eng['step']()

            if n % stride == 0:
                saved_steps[frame] = n
                epot[frame] = ens.energy.reshape(-1).detach().cpu().numpy()
                temp[frame] = temperature_per_system(mass_flat, ens.velocities.detach().cpu(), natoms)
                pos[frame] = ens.positions.detach().cpu().numpy()

                frame += 1
                stop_if_nan(n, names, epot[frame - 1], temp[frame - 1], split_by_system(pos[frame - 1], natoms))
                
                if frame == size:
                    write_fiumd_part(run_dir, composition, natoms, part, saved_steps, epot, temp, pos)
                    save_checkpoint(run_dir, 'checkpoint.npz', n, part, ens.positions, ens.velocities, device)
                    frame = 0
                    part += 1



            if n % progress == 0:
                print(f'fiumd: {n}/{steps} ({100 * n / steps:.0f} %)', flush=True)

    finally: 
        if frame > 0:
            write_fiumd_part(run_dir, composition, natoms, part, saved_steps[:frame],epot[:frame], temp[:frame], pos[:frame])
    

    save_checkpoint(run_dir, 'checkpoint.npz', steps, part, ens.positions, ens.velocities, device)
    merge_parts(run_dir)

def run_torchmd(engines, steps, stride, run_dir, save_every, device):

    '''
    One loop per species, one after the other
    '''
    size = min(save_every, steps // stride)
    progress = max(steps // 20, 1)

    for key, eng in engines.items():
        system = eng['system']
        k, natoms = system.pos.shape[:2]
        names = [f'{key} replica {i}' for i in range(k)]

        start, part = resume_point(run_dir, f'checkpoint_{key}.npz', [key], device)

        if (Path(run_dir) / f'traj_{key}.npz').exists():
            # A merge killed while deleting can leave parts behind
            drop_newer_parts(run_dir, [key], 0)
            continue


        saved_steps = np.zeros(size, dtype=int)
        dtype = system.pos.detach().cpu().numpy().dtype

        epot = np.zeros((size, k), dtype=dtype)
        temp = np.zeros((size, k), dtype=dtype)
        pos = np.zeros((size, k, natoms, 3), dtype=dtype)

        frame = 0
        try:
            for n in range(start + 1, steps + 1):

                _, pot, T = eng['integrator'].step(niter=1)

                if n % stride == 0:
                    saved_steps[frame] = n
                    epot[frame] = pot
                    temp[frame] = T
                    pos[frame] = system.pos.detach().cpu().numpy()

                    frame += 1
                    stop_if_nan(n, names, epot[frame - 1], temp[frame - 1], pos[frame - 1])

                    if frame == size:
                        write_part(run_dir, key, part, saved_steps, epot, temp, pos)
                        save_checkpoint(run_dir, f'checkpoint_{key}.npz', n, part, system.pos, system.vel, device)
                        frame = 0
                        part += 1



                if n % progress == 0:
                    print(f'torchmd {key}: {n}/{steps} ({100 * n / steps:.0f} %)', flush=True)
        finally:
            # Partial buffer. It also runs on a NaN, an exception or Ctrl-C
            if frame > 0:
                write_part(run_dir, key, part, saved_steps[:frame], epot[:frame], temp[:frame], pos[:frame])
        # Normal end only: a kill during the merge has nothing left to simulate
        save_checkpoint(run_dir, f'checkpoint_{key}.npz', steps, part, system.pos, system.vel, device)
        merge_parts(run_dir)


def save_trajectory(run_dir, key, steps, epot, temp, pos):
    '''
    steps (nframes,), epot and temp (nframes, k), pos (nframes, k, natoms, 3)
    '''
    write_npz_atomic(Path(run_dir) / f'traj_{key}.npz', steps=steps, epot=epot, temp=temp, pos=pos)


def write_npz_atomic(path, **arrays):
    '''
    Writes to a temporary file and renames it, so a kill never leaves a half-written npz
    '''
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def write_part(run_dir, key, part, steps, epot, temp, pos):
    '''
    One chunk of a trajectory, same arrays as traj_<key>.npz
    '''
    write_npz_atomic(Path(run_dir) / f'traj_{key}.part{part:04d}.npz', steps=steps, epot=epot, temp=temp, pos=pos)

def write_fiumd_part(run_dir, composition, natoms, part,steps,epot, temp, pos):
    
    offsets = np.cumsum(np.concatenate(([0], natoms)))
    
    for key in dict.fromkeys(composition):
        idx = [i for i, k in enumerate(composition) if k == key]
        pos_key = np.stack([pos[:, offsets[i]:offsets[i + 1]] for i in idx], axis=1)
    
        write_part(run_dir, key, part, steps, epot[:, idx], temp[:, idx], pos_key)

def merge_parts(run_dir):
    
    run_dir = Path(run_dir)
    parts_per_key = {}

    for path in run_dir.glob('traj_*.part*.npz'):
        key = path.name[len('traj_'):].split('.part')[0]
        parts_per_key.setdefault(key, []).append(path)

    for key, parts in parts_per_key.items():
       
        parts.sort(key=lambda path: int(path.name.split('.part')[1].split('.')[0]))
        final = run_dir / f'traj_{key}.npz'

        #If the final file exists, a previous merge finished and was killed while deleting
        if not final.exists():
            chunks = []
            for path in parts:
                with np.load(path) as z:
                    chunks.append({name: z[name] for name in z.files})

            steps = np.concatenate([c['steps'] for c in chunks])
            if not np.array_equal(steps, steps[0] * np.arange(1, len(steps) + 1)):
                sys.exit(f'{key}: there are missing frames in the parts, nothing was merged or deleted')

            save_trajectory(run_dir, key, steps,
                            np.concatenate([c['epot'] for c in chunks]),
                            np.concatenate([c['temp'] for c in chunks]),
                            np.concatenate([c['pos'] for c in chunks]))

            with np.load(final) as z:
                if len(z['steps']) != len(steps):
                    sys.exit(f'{final}: merge has {len(z["steps"])} frames, expected {len(steps)}. Parts were kept')

        for path in parts:
            path.unlink()

def drop_newer_parts(run_dir, keys, part):
    '''
    Deletes the parts after `part` and the temporary files of a killed write
    '''
    for key in keys:
        for path in Path(run_dir).glob(f'traj_{key}.part*.npz'):
            if int(path.name.split('.part')[1].split('.')[0]) > part:
                path.unlink()

    for path in Path(run_dir).glob('*.tmp'):
        path.unlink()

def save_checkpoint(run_dir, name, step, part, pos, vel, device):
    '''
    Restart state. Never call it from a finally: a NaN state would replace the last good one
    '''
    rng = dict(rng_cpu=torch.get_rng_state().numpy())
    if device.startswith('cuda'):
        rng['rng_gpu'] = torch.cuda.get_rng_state(device).numpy()

    write_npz_atomic(Path(run_dir) / name, step=step, part=part,
                     pos=pos.detach().cpu().numpy(), vel=vel.detach().cpu().numpy(), **rng)

def load_checkpoint(run_dir, name):

    path = Path(run_dir) / name
    if not path.exists():
        return None

    with np.load(path) as z:
        return {k: z[k] for k in z.files}
    
def resume_point(run_dir, name, keys, device):
    '''
    Returns the last finished step and the next part. Restores the RNG, so it must
    run after the manual_seed of main. Without a checkpoint the RNG is not touched
    '''
    start, last_part = 0, 0

    ckpt = load_checkpoint(run_dir, name)
    if ckpt is not None:
        start, last_part = int(ckpt['step']), int(ckpt['part'])

        torch.set_rng_state(torch.from_numpy(ckpt['rng_cpu']))
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(torch.from_numpy(ckpt['rng_gpu']), device)

    drop_newer_parts(run_dir, keys, last_part)

    return start, last_part + 1

def check_resume(args, run_dir):
    '''
    Stops if the run is not resumable or if this command does not match its config.json
    '''
    path = Path(run_dir) / 'config.json'
    if not path.exists():
        sys.exit(f'--resume: {path} does not exist')
    if (Path(run_dir) / 'COMPLETED').exists():
        sys.exit(f'--resume: {run_dir} is already complete')

    with open(path) as f:
        config = json.load(f)

    diff = [f'{name}: run has {config.get(name)}, command has {getattr(args, name)}'
            for name in RESUME_CHECK if config.get(name) != getattr(args, name)]
    if diff:
        sys.exit('--resume: arguments differ from config.json\n  ' + '\n  '.join(diff))

def stop_on_sigterm(*_):

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.exit(143)

def write_config (args, run_dir):
    import subprocess
    import fiumd

    config = dict (vars(args))
    config['torch_version'] = torch.__version__
    config['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    config['fiumd_commit'] = subprocess.run(['git', '-C', str(Path(fiumd.__file__).parent), 'rev-parse', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip() or None

    with open (Path(run_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent = 2, default = str)
if __name__ == "__main__":

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    #GENERAL PARAMETERS 
    p.add_argument("--composition", default="cln,trp,bbl,lambda", help="molecules separated by commas, e.g. cln:2,trp,bbl")
    p.add_argument("--temps", default="" , help="T by system for examplecln:300 ,trp:350")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--initial_state", default=None,help='npz file with the initial state to read; if not given, fiumd samples it and saves it in <outdir>/<run_id>/initial_state.npz' )
    p.add_argument('--precision', choices=list(PRECISIONS), default='float32')


    #FOLDERS 
    p.add_argument("--run_id", required=True, help="the run is saved in <outdir>/<run_id>")

    p.add_argument("--datos_dir", default=str(REPO_ROOT / "datos"))
    p.add_argument("--outdir", default=str(REPO_ROOT / "results" / "sistemas_multiples"))
    p.add_argument("--temp", type=float, default=350)

    p.add_argument('--save_every', type=int, default=500, help='frames between writes to disk')
    p.add_argument('--resume', action = 'store_true',  help='continue <outdir>/<run_id> from its last checkpoint')
    #SIMULATION PARAMETERS

    p.add_argument("--engine",choices=['fiumd', 'torchmd'],default ='fiumd', help="Select between the two supported engines fiumd or torchmd")

    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--stride', type=int, required=True)
    p.add_argument('--dt', type=float, default=1.0, help = 'timestep in fs')
    p.add_argument('--gamma', type=float, default=1.0, help='unit ps**(-1)')
    p.add_argument('--dts', default='', help='dt by species in fs, for example bbl:0.5,lambda:0.5; the rest use --dt')
    p.add_argument('--gammas', default='', help='gamma by species in ps**(-1), for example cln:0.5; the rest use --gamma')



    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--nnp', choices=['cln', 'bbl', 'lambda'], required=True, help='species whose checkpoint is used for the whole system')


    args = p.parse_args()
    signal.signal(signal.SIGTERM,stop_on_sigterm)

    seed = args.seed 
    datos_dir = args.datos_dir
    engine = args.engine
    dtype = PRECISIONS[args.precision]

    if args.stride <= 0 or args.steps < args.stride:
        sys.exit(f'--stride must be > 0 and <= --steps, got steps = {args.steps}, stride = {args.stride}')

    if args.save_every <= 0:
        sys.exit(f'--save_every must be > 0, got {args.save_every}')

    run_dir = Path(args.outdir) / args.run_id
    if args.resume:
        check_resume(args, run_dir)


    composition = parse_composition(args.composition)

    commons = load_molecules(composition, datos_dir)

    temps = parse_temps(args.temps, composition, args.temp)

    dts = parse_temps(args.dts, composition, args.dt, '--dts')
    gammas = parse_temps(args.gammas, composition, args.gamma, '--gammas')

    initial_state = args.initial_state
    if args.resume and (run_dir / 'initial_state.npz').exists():
        initial_state = run_dir / 'initial_state.npz'

    if initial_state:
        state = load_state(initial_state, composition, temps, commons)
    elif engine == 'torchmd':
        sys.exit('torchmd does not sample its own state: pass --initial_state with the npz from fiumd')
    else:
        state = make_initial_state(commons, composition, temps, seed)

    check_initial_temperatures(state['vel_flat'], commons, composition, temps)
    for key in dict.fromkeys(composition):
        pos, _ = molecule_state(state, key)
        check_bonds(pos, commons[key]['mol'])

    # The run folder is created after validating, so a failed start does not block the run_id
    if not args.resume:
        make_run_dir(args.outdir, args.run_id)
        write_config(args, run_dir)
    if not initial_state:
        save_state(run_dir / 'initial_state.npz', state)

    nnp_model = load_nnp(args.nnp, commons, datos_dir, args.device, derivative = engine == 'torchmd', dtype=dtype)

    if engine == 'fiumd':
        ckpt = load_checkpoint(run_dir, 'checkpoint.npz')
        if ckpt is not None:
            state['pos_flat'] = torch.from_numpy(ckpt['pos'])
            state['vel_flat'] = torch.from_numpy(ckpt['vel'])

        # One value per system: the integrators take tensors of shape (nsystems,)
        dt = torch.tensor([dts[k] for k in composition], dtype=torch.float64)
        gamma = torch.tensor([gammas[k] for k in composition], dtype=torch.float64)
        eng = build_fiumd(commons, state, gamma, dt, args.device, nnp_model, dtype)
    else:
        engines = {}
        for key in dict.fromkeys(composition):
            pos, vel = molecule_state(state, key)
            
            ckpt = load_checkpoint(run_dir, f'checkpoint_{key}.npz')
            if ckpt is not None:
                # build_torchmd takes positions as (natoms, 3, k)
                pos = np.moveaxis(ckpt['pos'], 0, -1)
                vel = torch.from_numpy(ckpt['vel'])

            engines[key] = build_torchmd(commons[key], pos, vel, temps[key], gammas[key], dts[key], args.device, nnp_model, dtype)


    torch.manual_seed(seed + 1)   

    if engine == 'fiumd':
        run_fiumd(eng, state, args.steps, args.stride, run_dir, args.save_every, args.device)
    else:
        run_torchmd(engines, args.steps, args.stride, run_dir, args.save_every, args.device)


    # Only reached after the merge: a dead run never has this file
    (run_dir / 'COMPLETED').touch()
