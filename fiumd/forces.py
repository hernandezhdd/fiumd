from __future__ import annotations
import torch
import torch.nn as nn
import fiumd.systems
import fiumd.utils

class Force(nn.Module):
    """
    Abstract base class for a force term (bond, angle, torsion, etc).
    Each subclass must implement the compute() method.
    """
    registry = {}
    
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Force.registry[cls.__name__] = cls
        
    @classmethod
    def build_params(cls, pairs, system, **kwargs):
        raise NotImplementedError("Subclasses must implement build_params()")

    def forward(self, ensemble, inplace=True):
        """
        Compute energy and forces for this term.

        Parameters
        ----------
        ensemble : Ensemble
            An ensemble object containing current positions and precomputed quantities.

        Returns
        -------
        energy : torch.Tensor
            Scalar or (n_terms,) energy tensor.
        forces : torch.Tensor
            Tensor of shape (n_atoms, 3) with forces to apply.
        """
        raise NotImplementedError("Subclasses must implement compute()")

class BondForce(Force):
    def __init__(self,
                 force_idx_map: torch.Tensor,
                 global_pair_idx: torch.Tensor,
                 pair_to_system: torch.Tensor,
                 atom_types: torch.Tensor,
                 kwarg_params: dict[str, torch.tensor],
                 # inplace=True,
                 # explicit_forces=True,
                ):
        """
        Initialize the BondForce object.

        Parameters:
        - force_idx_map (Tensor): Maps bond pairs to global pairs indices.
        - force_params (Tensor): Parameters for each bond (e.g., force constants, equilibrium distances).
        - global_pair_idx (Tensor): List of all unique atom pairs involved in bond interactions globally.
        - pair_to_system (Tensor): Maps each global bond pair to its system index (which system it belongs to).
        - inplace (bool): If True (default), update the ensemble's energy and forces in-place.
                          If False, do NOT modify the ensemble, but return computed tensors instead.
        - explicit_forces (bool): Request explicit forces to be returned
        """
        super().__init__("Bond")
        self.force_idx_map = nn.parameter.Buffer(force_idx_map.to(torch.long), persistent=True)
        self.r0 = nn.parameter.Buffer(kwarg_params['params'][:, 1], persistent=True)
        self.k = nn.parameter.Buffer(kwarg_params['params'][:, 0], persistent=True)
        self.global_pair_idx = nn.parameter.Buffer(global_pair_idx, persistent=True)
        self.pair_to_system = nn.parameter.Buffer(pair_to_system, persistent=True)

        # self.inplace = inplace
        # self.explicit_forces = explicit_forces
        
        # Derived indices
        self.pair_atoms = nn.parameter.Buffer(self.global_pair_idx[self.force_idx_map], persistent=True)
        self.pot = nn.parameter.Buffer( torch.empty(force_idx_map.shape), persistent=False)
        self.x = nn.parameter.Buffer( torch.empty(force_idx_map.shape), persistent=False)
        self.force = nn.parameter.Buffer( torch.empty(force_idx_map.shape[0],3), persistent=False)

    @classmethod
    def build_params(cls,
                     system: fiumd.systems.System,
                     ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        system : fiumd.systems.System
    
        Returns
        -------
        kwarg_params : dict[str, torch.Tensor]
        """
        kwarg_params = system.params.bond_params
    
        return kwarg_params

    def forward(self, ensemble, inplace=True):
        """
        Parameters
        ----------
        all_distances : torch.Tensor
            Shape: (npairs_all,)
        all_unitvec : torch.Tensor
            Shape: (npairs_all, 3)
        force_idx_map : torch.Tensor
            Shape: (nbonds,)
        force_params : torch.Tensor
            Shape: (nbonds, 2), with [k, r0]
        explicit_forces : bool
            Whether to compute force vectors
    
        Returns
        -------
        pot : torch.Tensor
            Shape: (nbonds,)
        force : torch.Tensor or None
            Shape: (nbonds, 3) or
        """
        
        # Evaluate bond potentials and forces using distances and unit vectors from the ensemble
        self.x.copy_(ensemble.pairs_distances[self.force_idx_map]) # (nbonds,)
        self.x.sub_(self.r0)

        # self.pot initially holds k*x, becomes k*x^2 after mul_
        self.pot.copy_(self.x)        # (nbonds,)
        self.pot.mul_(self.k)
        
        self.force.copy_(ensemble.pairs_unitvec[self.force_idx_map])      # (nbonds, 3)
        self.force.mul_( 2 * self.pot.unsqueeze(-1) ) # here self.pot is just k*x

        self.pot.mul_(self.x)


        
        if inplace:
            # Accumulate bond potential energy into the ensemble's energy tensor,
            # indexing energy by the system the bond belongs to
            ensemble.energy.index_add_(0, self.pair_to_system[self.force_idx_map], self.pot)

            # Accumulate forces on the atoms involved in each bond:
            # For the first atom in the pair, subtract the bond force vector
            ensemble.forces.index_add_(0, self.pair_atoms[:, 0], -self.force)

            # For the second atom in the pair, add the bond force vector
            ensemble.forces.index_add_(0, self.pair_atoms[:, 1], +self.force)

            # No return needed since ensemble is updated in-place
            return None
        else:
            # Return computed potentials and bond forces without modifying ensemble

            # Accumulate bond potential energy into the ensemble's energy tensor,
            # indexing energy by the system the bond belongs to
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.pair_to_system[self.force_idx_map], self.pot)

            # Accumulate forces on the atoms involved in each bond:
            # For the first atom in the pair, subtract the bond force vector
            forces.index_add_(0, self.global_pair_idx[self.force_idx_map][:, 0], -self.force)

            # For the second atom in the pair, add the bond force vector
            forces.index_add_(0, self.global_pair_idx[self.force_idx_map][:, 1], +self.force)
            return energy, forces
            
    def info(self):
        fiumd.utils.info(self)


class RepulsionCGForce(Force):
    def __init__(self,
                 force_idx_map: torch.Tensor,
                 global_pair_idx: torch.Tensor,
                 pair_to_system: torch.Tensor,
                 atom_types: torch.Tensor,
                 kwarg_params: list[dict[str, torch.Tensor]],
                 scale=1.0,
                 # inplace=True,
                 # explicit_forces=True,
                ):
        """
        Repulsion force using a coarse-grained repulsion model.

        Parameters:
        - force_idx_map (Tensor): Maps the relevant repulsion pair indices to the global pair list.
        - global_pair_idx (Tensor): Atom index pairs (N, 2).
        - pair_to_system (Tensor): Maps global pair to system index.
        - atom_types (Tensor): Atom type for each atom (int tensor of shape (n_atoms,)).
        - B (Tensor): (ntypes x ntypes) repulsion coefficient matrix.
        - scale (float): Scaling factor for energy/force.
        """
        super().__init__("RepulsionCG")

        # Persistent buffers
        self.force_idx_map = nn.parameter.Buffer(force_idx_map, persistent=True)
        self.global_pair_idx = nn.parameter.Buffer(global_pair_idx, persistent=True)
        self.pair_to_system = nn.parameter.Buffer(pair_to_system, persistent=True)
        self.atom_types = nn.parameter.Buffer(atom_types, persistent=True)
        self.B = nn.parameter.Buffer(kwarg_params['B'], persistent=True)
        self.scale = scale

        # Derived indices
        self.pair_atoms = nn.parameter.Buffer(self.global_pair_idx[self.force_idx_map], persistent=True)
        self.type_i = nn.parameter.Buffer(self.atom_types[self.pair_atoms[:, 0]], persistent=True)
        self.type_j = nn.parameter.Buffer(self.atom_types[self.pair_atoms[:, 1]], persistent=True)
        self.coef = nn.parameter.Buffer(self.B[self.type_i, self.type_j], persistent=True)   # (npairs,)

        # Temporary buffers (non-persistent)
        npairs = self.force_idx_map.shape[0]
        self.pot = nn.parameter.Buffer(torch.empty(npairs), persistent=False)
        self.force = nn.parameter.Buffer(torch.empty((npairs, 3)), persistent=False)
        self.rinv1 = nn.parameter.Buffer(torch.empty(npairs), persistent=False)
        self.rinv6 = nn.parameter.Buffer(torch.empty(npairs), persistent=False)

    @classmethod
    def build_params(cls,
                     system: fiumd.systems.System,
                     ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        system : fiumd.systems.System
    
        Returns
        -------
        nonbonded_params : dict[str, torch.Tensor]
        """
        kwarg_params = {}
        
        kwarg_params = system.params.nonbonded_params

        _, kwarg_params['B'] = system.params.get_AB()
    
        return kwarg_params
    
    def forward(self, ensemble, inplace=True):
        """
        Compute repulsion CG potentials and forces.

        Parameters:
        - ensemble: The Ensemble object containing distance and direction data.
        - inplace: Whether to write into the ensemble energy/force tensors directly.

        Returns:
        - None if inplace=True
        - Tuple (energy, forces) if inplace=False
        """
        self.rinv1.copy_(ensemble.pairs_distances[self.force_idx_map])  # (npairs,)
        self.rinv1.reciprocal_()                                       # 1/r
        self.rinv6.copy_(self.rinv1).pow_(6)                           # 1/r^6

        self.pot.copy_(self.coef).mul_(self.rinv6).div_(self.scale)   # V_CG = B * (1/r)^6 / scale

        # F = -6 * B / r^7 / scale = -6 * V_CG / r * unitvec
        self.force.copy_(ensemble.pairs_unitvec[self.force_idx_map])
        self.force.mul_(-6).mul_(self.pot.unsqueeze(-1)).mul_(self.rinv1.unsqueeze(-1))

        if inplace:
            ensemble.energy.index_add_(0, self.pair_to_system[self.force_idx_map], self.pot)
            ensemble.forces.index_add_(0, self.pair_atoms[:, 0], -self.force)
            ensemble.forces.index_add_(0, self.pair_atoms[:, 1], +self.force)
            return None
        else:
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.pair_to_system[self.force_idx_map], self.pot)
            forces.index_add_(0, self.pair_atoms[:, 0], -self.force)
            forces.index_add_(0, self.pair_atoms[:, 1], +self.force)
            return energy, forces


class TorchMDNetForce(Force):
    def __init__(self,
                 model: nn.Module,
                 embeddings: torch.Tensor,
                 batch: torch.Tensor,
                 # output_transform=None,
                 dtype=torch.float32,
                 device="cpu",
                 box=None, 
                ):

        super().__init__("TorchMDNet")
        self.model = model.eval().to(device)
        # self.device = device
        self.dtype = dtype
        # box = getattr(ensemble, "box", None)

        self.box = nn.parameter.Buffer(box.to(self.dtype)) if box is not None else None

        # self.output_transformer = (
        #     lambda e, f: (e, f) if output_transform is None else output_transform
        # )

        self.embeddings = nn.parameter.Buffer(embeddings)
        self.batch = nn.parameter.Buffer(batch)

        # self.pot = nn.parameter.Buffer(torch.empty(npairs), persistent=False)
        # self.force = nn.parameter.Buffer(torch.empty((npairs, 3)), persistent=False)

    def forward(self, ensemble, inplace=True):
        

        energy, forces = self.model(self.embeddings,
                                    ensemble.positions,
                                    self.batch, self.box)
        # energy, forces = self.output_transformer(
        #     energy.detach(),
        #     forces.detach().reshape(-1, self.n_atoms, 3)
        # )

        if inplace:
            ensemble.energy.add_(energy.squeeze())
            ensemble.forces.add_(forces)
            return None
        else:
            return energy, forces

            
    def info(self):
        fiumd.utils.info(self)

class DihedralForce(Force):
    def __init__(self,
                 force_idx_map: torch.Tensor,
                 global_dihed_term_idx: torch.Tensor,
                 dihed_term_system_idx: torch.Tensor,
                 torsion_param_map: torch.Tensor,
                 kwarg_params: dict[str, torch.Tensor],
                 ):
        super().__init__("Dihedral")

        self.dihedral_idx_map = nn.parameter.Buffer(force_idx_map, persistent=True)
        self.global_dihed_term_idx = nn.parameter.Buffer(global_dihed_term_idx, persistent=True)
        self.dihed_term_system_idx = nn.parameter.Buffer(dihed_term_system_idx, persistent=True)
        self.torsion_param_map = nn.parameter.Buffer(torsion_param_map, persistent=True)

        # Static buffers for k, phi0, periodicity
        self.k = nn.parameter.Buffer(kwarg_params['params'][:, 0], persistent=True)
        self.phi0 = nn.parameter.Buffer(kwarg_params['params'][:, 1], persistent=True)
        self.periodicity = nn.parameter.Buffer(kwarg_params['params'][:, 2], persistent=True)

        # Preallocate intermediate results
        nterms = torsion_param_map.shape[0]
        self.pot = nn.parameter.Buffer(torch.empty(nterms), persistent=False)
        self.force0 = nn.parameter.Buffer(torch.empty(nterms, 3), persistent=False)
        self.force1 = nn.parameter.Buffer(torch.empty(nterms, 3), persistent=False)
        self.force2 = nn.parameter.Buffer(torch.empty(nterms, 3), persistent=False)
        self.force3 = nn.parameter.Buffer(torch.empty(nterms, 3), persistent=False)

    @classmethod
    def build_params(cls, system):
        return system.params.dihedral_params

    def forward(self, ensemble, inplace=True):

        evaluate_torsion_into(
            self.pot,
            self.force0, self.force1, self.force2, self.force3,
            all_relative_positions=ensemble.pairs_relative_positions,
            all_distances=ensemble.pairs_distances,
            dihedral_idx_map=self.force_idx_map,
            torsion_param_map=self.torsion_param_map,
            k=self.k,
            phi0=self.phi0,
            periodicity=self.periodicity,
        )

        if inplace:
            ensemble.energy.index_add_(0, self.dihed_term_system_idx, self.pot)
            ensemble.forces.index_add_(0, self.global_dihed_term_idx[:, 0], self.force0)
            ensemble.forces.index_add_(0, self.global_dihed_term_idx[:, 1], self.force1)
            ensemble.forces.index_add_(0, self.global_dihed_term_idx[:, 2], self.force2)
            ensemble.forces.index_add_(0, self.global_dihed_term_idx[:, 3], self.force3)
            return None
        else:
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.dihed_term_system_idx, self.pot)
            forces.index_add_(0, self.global_dihed_term_idx[:, 0], self.force0)
            forces.index_add_(0, self.global_dihed_term_idx[:, 1], self.force1)
            forces.index_add_(0, self.global_dihed_term_idx[:, 2], self.force2)
            forces.index_add_(0, self.global_dihed_term_idx[:, 3], self.force3)
            return energy, forces


def evaluate_torsion_into(
    pot: torch.Tensor,
    force0: torch.Tensor,
    force1: torch.Tensor,
    force2: torch.Tensor,
    force3: torch.Tensor,
    *,
    all_relative_positions: torch.Tensor,
    all_distances: torch.Tensor,
    dihedral_idx_map: torch.Tensor,
    torsion_param_map: torch.Tensor,
    k: torch.Tensor,
    phi0: torch.Tensor,
    periodicity: torch.Tensor,
):
    dih_idx = torsion_param_map[:, 0]

    r12 = all_relative_positions[dihedral_idx_map[dih_idx, 0]]
    r23 = all_relative_positions[dihedral_idx_map[dih_idx, 1]]
    r34 = all_relative_positions[dihedral_idx_map[dih_idx, 2]]

    crossA = torch.cross(r12, r23, dim=1)
    crossB = torch.cross(r23, r34, dim=1)
    crossC = torch.cross(r23, crossA, dim=1)

    normA = 1e-8 + crossA.norm(dim=1)
    normB = 1e-8 + crossB.norm(dim=1)
    normC = 1e-8 + crossC.norm(dim=1)

    crossB_normed = crossB / normB.unsqueeze(1)
    cos_phi = (crossA * crossB_normed).sum(dim=1) / normA
    sin_phi = (crossC * crossB_normed).sum(dim=1) / normC
    phi = -torch.atan2(sin_phi, cos_phi)

    is_amber = (periodicity > 0).all()

    if is_amber:
        angle_diff = periodicity * phi - phi0
        pot.copy_(k * (1 + torch.cos(angle_diff)))
        coeff = -periodicity * k * torch.sin(angle_diff)
    else:
        angle_diff = (phi - phi0 + math.pi) % (2 * math.pi) - math.pi
        pot.copy_(k * angle_diff**2)
        coeff = 2 * k * angle_diff

    norm_r23 = all_distances[dihedral_idx_map[dih_idx, 1]]
    norm_r23_sq = norm_r23**2 + 1e-8

    dot12 = (r12 * r23).sum(dim=1)
    dot34 = (r34 * r23).sum(dim=1)

    force_factor0 = (-coeff * norm_r23) / (normA**2)
    force_factor1 = dot12 / norm_r23_sq
    force_factor2 = dot34 / norm_r23_sq
    force_factor3 = (coeff * norm_r23) / (normB**2)

    force0vec = force_factor0.unsqueeze(1) * crossA
    force3vec = force_factor3.unsqueeze(1) * crossB
    s = force_factor1.unsqueeze(1) * force0vec - force_factor2.unsqueeze(1) * force3vec

    force0.copy_(-force0vec)
    force1.copy_(force0vec + s)
    force2.copy_(force3vec - s)
    force3.copy_(-force3vec)