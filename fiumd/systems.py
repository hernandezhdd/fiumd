import torch
import torch.nn as nn
# from fiumd.forces import Force
import fiumd.forces
import fiumd.utils
import fiumd.parameters
import numpy as np
from typing import Type

class System(nn.Module):
    def __init__(self,
                 force_terms: list[str],
                 # params: dict[str, dict[str, torch.Tensor]],
                 params: fiumd.parameters.Parameters,
                 box,
                ):
        """
        Parameters
        ----------

        box : torch.Tensor, shape (3,)
        force_terms : force term present. list of str with the names of fiumd.forces.Force subclasses
            For example: {'BondForce', RepulsionCGForce}
        params: fiumd.parameters.Parameters
        """
        super().__init__()

        self.box = nn.parameter.Buffer(box)               # (3,)
        self.force_terms = force_terms  # per-interaction type
        self.params = params
        self.natoms = len(params.masses)
        self.params_by_force = self._build_force_params()

    def __repr__(self):
        return f"<System: {self.natoms} atoms>"

        
    def _build_force_params(self):

        params_by_force = {}

        for force_name in self.force_terms:

            force_cls = fiumd.forces.Force.registry[force_name]
            # kwargs = self.params[force_name]

            params = force_cls.build_params(self)# , **kwargs)
            
            params_by_force[force_name] = params

        return params_by_force
    

    def info(self):
        fiumd.utils.info(self)
        
# class System:
#     def __init__(self, subsystems: list[SubSystem],
#                  precision: torch.dtype, device: torch.device):

#         self.n_subsystems = len(subsystems)

#         self.box = torch.zeros(nreplicas, 3, 3)
#         self.pos, self.vel, self.forces = concat_subsystems(subsystems)

#         self.to_(device)
#         self.precision_(precision)

#     # @property
#     # def natoms(self):
#     #     return self.pos.shape[1]

#     # @property
#     # def n_subsystems(self):
#     #     return self.pos.shape[0]

#     def to_(self, device):
#         self.forces = self.forces.to(device)
#         self.box = self.box.to(device)
#         self.pos = self.pos.to(device)
#         self.vel = self.vel.to(device)

#     def precision_(self, precision):
#         self.forces = self.forces.type(precision)
#         self.box = self.box.type(precision)
#         self.pos = self.pos.type(precision)
#         self.vel = self.vel.type(precision)


#     def set_box(self, box):
#         if box.ndim == 1:
#             if len(box) != 3:
#                 raise RuntimeError("Box must have at least 3 elements")
#             box = box[:, None]

#         if box.shape[0] != 3:
#             raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

#         box = np.swapaxes(box, 1, 0)

#         if self.nreplicas > 1 and box.shape[0] != self.nreplicas:
#             box = np.repeat(box[0][None, :], self.nreplicas, axis=0)

#         for r in range(box.shape[0]):
#             self.box[r][torch.eye(3).bool()] = torch.tensor(
#                 box[r], dtype=self.box.dtype, device=self.box.device
#             )

#     def set_forces(self, forces):
#         if forces.shape != (self.nreplicas, self.natoms, 3):
#             raise RuntimeError("Forces shape must be (nreplicas, natoms, 3)")
#         self.forces[:] = torch.tensor(
#             forces, dtype=self.forces.dtype, device=self.forces.device
#         )

        
# class SubSystem:
#     def __init__(self, natoms: int, force_terms: list[Force]):
#         self.natoms = natoms
#         self.force_terms = force_terms

#         # Mutable state
#         self.positions = None         # (natoms, 3)
#         self.velocities = None        # Optional (natoms, 3)
#         self.forces = None        # Optional (natoms, 3)
#         self.masses = None            # Optional (natoms,)
        
#         self.all_relative_positions = None  # (n_pairs, 3)
#         self.all_distances = None          # (n_pairs,)

#     def set_positions(self, positions: torch.Tensor):
#         """Set positions and update geometry."""
#         self.positions = positions
#         self.update_geometry()

#     def set_velocities(self, velocities: torch.Tensor):
#         self.velocities = velocities

#     def set_masses(self, masses: torch.Tensor):
#         self.masses = masses

#     def update_geometry(self):
#         """Update all_relative_positions and all_distances from positions."""
#         # This needs pair index mapping available
#         assert hasattr(self, 'pair_indices'), "Must set pair_indices before updating geometry"
#         ri = self.positions[self.pair_indices[:, 0]]
#         rj = self.positions[self.pair_indices[:, 1]]
#         self.all_relative_positions = rj - ri
#         self.all_distances = self.all_relative_positions.norm(dim=1)

#     def set_pair_indices(self, pair_indices: torch.Tensor):
#         """Used to compute displacements/distances (e.g., for torsions)."""
#         self.pair_indices = pair_indices

#     def compute_forces(self):
#         total_energy = 0.0
#         total_forces = torch.zeros_like(self.positions)

#         for force in self.forces:
#             energy, forces = force.compute(self)
#             total_energy += energy
#             total_forces += forces

#         return total_energy, total_forces

# def concat_subsystems(subsystems: list[SubSystem]):
#     pos = []
#     vel = []
#     forces = []
#     for subsys in subsystems:
#         pos.append(subsys.positions)
#         vel.append(subsys.velocities)
#         forces.append(subsys.forces)

#     pos = torch.cat(pos, dim=0).unsqueeze(0)
#     vel = torch.cat(vel, dim=0).unsqueeze(0)
#     forces = torch.cat(forces, dim=0).unsqueeze(0)

#     return pos, vel, forces
#     # return {'pos': pos_list, 'vel': vel_list, 'forces': forces_list}