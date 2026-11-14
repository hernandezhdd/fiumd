"""Minimal FiuMD example: chignolin and Trp-cage advanced in one batch.

Classical priors + one shared TorchMD-Net correction, Velocity Verlet with a
Langevin thermostat, a different temperature per protein.

Needs the inputs of paper/datos (see paper/datos/README.md):

    cd paper && python datos/download_data.py --molecules cln trp lambda \
        && python benchmark/check_benchmark_data.py --fix
    python examples/two_proteins.py --datos paper/datos
"""
import argparse

import numpy as np
import torch
from moleculekit.molecule import Molecule
from torchmdnet.models.model import load_model

from fiumd.ensembles import Ensemble
from fiumd.forcefields.forcefield import ForceField
from fiumd.integrator import LangevinThermostat, VelocityVerletIntegrator
from fiumd.parameters import Parameters
from fiumd.systems import System

BOLTZMANN = 0.001987191  # kcal/mol/K
FORCE_TERMS = {"BondForce", "RepulsionCGForce", "DihedralForce"}
PARAM_TERMS = ["Bonds", "RepulsionCG", "Dihedrals"]
# CA bead type -> embedding of the checkpoints of Majewski et al. (2023)
EMBEDDINGS = {"CAA": 1, "CAG": 2, "CAF": 3, "CAY": 4, "CAD": 5, "CAE": 6, "CAW": 7, "CAP": 8, "CAN": 9, "CAQ": 10,
              "CAH": 11, "CAS": 12, "CAT": 13, "CAV": 14, "CAM": 15, "CAR": 19, "CAK": 20, "CAL": 21, "CAI": 22}

# protein -> (box side in A, target temperature in K)
PROTEINS = {"cln": (30.0, 300.0), "trp": (40.0, 350.0)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datos", default="paper/datos")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=1.0, help="fs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dtype, device = torch.float32, args.device
    torch.manual_seed(0)

    systems, molecules, temps = [], [], []
    for key, (side, temp) in PROTEINS.items():
        mol = Molecule(f"{args.datos}/{key}/{key}.psf")
        mol.read(f"{args.datos}/{key}/{key}_kcenters_32clusters_coords.xtc")
        mol.dropFrames(keep=0)  # first of the 32 starting structures
        ff = ForceField.create(mol, f"{args.datos}/{key}/{key}_prior_force_field.yaml")
        params = Parameters(ff, mol, PARAM_TERMS, precision=dtype, device=device)
        box = torch.full((3,), side, dtype=dtype, device=device)
        systems.append(System(molecule=mol, force_terms=FORCE_TERMS, params=params, box=box))
        molecules.append(mol)
        temps.append(temp)

    # One network for both proteins, energies only: FiuMD takes the gradient itself
    model = load_model(f"{args.datos}/lambda/model_modified.ckpt", derivative=False,
                       max_num_neighbors=torch.tensor(max(m.numAtoms for m in molecules), device=device))
    bead_ids = torch.tensor([EMBEDDINGS[t] for m in molecules for t in m.atomtype], device=device)

    ensemble = Ensemble(systems, device=device, dtype=dtype, nnp_model=model.to(device, dtype=dtype),
                        nnp_embeddings=bead_ids)
    ensemble.to(device).eval()

    # Positions and velocities of all beads, in system order: (N_total, 3)
    pos = torch.tensor(np.concatenate([m.coords[:, :, 0] for m in molecules]), dtype=dtype, device=device)
    t_per_bead = torch.cat([torch.full((m.numAtoms,), t) for m, t in zip(molecules, temps)]).to(device, dtype)
    masses = ensemble.masses.reshape(-1, 1)
    vel = torch.sqrt(BOLTZMANN * t_per_bead.unsqueeze(1) / masses) * torch.randn_like(pos)
    ensemble.set_positions(pos)  # also computes the initial forces
    ensemble.set_velocities(vel)

    dt = torch.full((len(systems),), args.dt, dtype=dtype, device=device)
    verlet = VelocityVerletIntegrator(dt=dt, ensemble=ensemble).to(device).eval()
    thermostat = LangevinThermostat(gamma=1.0, temperature=torch.tensor(temps, dtype=dtype, device=device),
                                    dt=dt, ensemble=ensemble).to(device).eval()

    for step in range(1, args.steps + 1):
        verlet()
        thermostat()
        if step % (args.steps // 10) == 0:
            ekin = 0.5 * (masses * ensemble.velocities ** 2).sum(-1)
            t_kin = [float(2 * e.sum() / (3 * len(e) * BOLTZMANN))
                     for e in ensemble.split_by_system(ekin)]
            print(f"step {step:5d}  E_pot " + "  ".join(f"{e:9.2f}" for e in ensemble.energy.tolist())
                  + "  kcal/mol   T_kin " + "  ".join(f"{t:6.1f}" for t in t_kin) + "  K")

    final = ensemble.split_by_system(ensemble.positions)
    print("final positions:", [tuple(x.shape) for x in final])


if __name__ == "__main__":
    main()