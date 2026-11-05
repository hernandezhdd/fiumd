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
                 term_to_pair: torch.Tensor,
                 global_atom_idx: torch.Tensor,
                 term_to_system: torch.Tensor,
                 term_to_param_idx: torch.Tensor,
                 atom_types: torch.Tensor,
                 kwarg_params: dict[str, torch.Tensor]):
        """
        Initialize the BondForce object.

        Parameters:
        - term_to_pair (Tensor): Maps bond pairs to global pairs indices.
        - force_params (Tensor): Parameters for each bond (e.g., force constants, equilibrium distances).
        - global_atom_idx (Tensor): List of all unique atom pairs involved in bond interactions globally.
        - term_to_system (Tensor): Maps each global bond pair to its system index (which system it belongs to).
        - inplace (bool): If True (default), update the ensemble's energy and forces in-place.
                          If False, do NOT modify the ensemble, but return computed tensors instead.
        - explicit_forces (bool): Request explicit forces to be returned
        """
        super().__init__("Bond")

        self.term_to_pair = nn.parameter.Buffer(term_to_pair, persistent=True)
        self.global_atom_idx = nn.parameter.Buffer(global_atom_idx, persistent=True)
        self.term_to_system = nn.parameter.Buffer(term_to_system, persistent=True)
        self.term_to_param_idx = nn.parameter.Buffer(term_to_param_idx, persistent=True) if term_to_param_idx is not None else None

        self.r0 = nn.parameter.Buffer(kwarg_params['params'][:, 1], persistent=True)
        self.k = nn.parameter.Buffer(kwarg_params['params'][:, 0], persistent=True)
        # self.inplace = inplace
        # self.explicit_forces = explicit_forces
        
        # Temporary buffers (non-persistent) matching parameter device and dtype
        nterms = self.term_to_pair.shape[0]
        buf_device = self.k.device
        buf_dtype = self.k.dtype
        self.pot = nn.parameter.Buffer(torch.empty(nterms, device=buf_device, dtype=buf_dtype), persistent=False)
        self.x = nn.parameter.Buffer(torch.empty(nterms, device=buf_device, dtype=buf_dtype), persistent=False)
        self.forces = nn.parameter.Buffer(torch.empty((nterms, 3), device=buf_device, dtype=buf_dtype), persistent=False)

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
        term_to_pair : torch.Tensor
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
        with torch.inference_mode():
            self.x.copy_(ensemble.pairs_distances[self.term_to_pair]) # (nbonds,)
            self.x.sub_(self.r0)
    
            # self.pot initially holds k*x, becomes k*x^2 after mul_
            self.pot.copy_(self.x)        # (nbonds,)
            self.pot.mul_(self.k)
            
            self.forces.copy_(ensemble.pairs_unitvec[self.term_to_pair])      # (nbonds, 3)
            self.forces.mul_( 2 * self.pot.unsqueeze(-1) ) # here self.pot is just k*x
    
            self.pot.mul_(self.x)
        
        if inplace:
            # Accumulate bond potential energy into the ensemble's energy tensor,
            # indexing energy by the system the bond belongs to
            ensemble.energy.index_add_(0, self.term_to_system, self.pot)

            # Accumulate forces on the atoms involved in each bond:
            # For the first atom in the pair, subtract the bond force vector
            ensemble.forces.index_add_(0, self.global_atom_idx[:, 0], -self.forces)

            # For the second atom in the pair, add the bond force vector
            ensemble.forces.index_add_(0, self.global_atom_idx[:, 1], +self.forces)

            # No return needed since ensemble is updated in-place
            return None
        else:
            # Return computed potentials and bond forces without modifying ensemble

            # Accumulate bond potential energy into the ensemble's energy tensor,
            # indexing energy by the system the bond belongs to
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.term_to_system, self.pot)

            # Accumulate forces on the atoms involved in each bond:
            # For the first atom in the pair, subtract the bond force vector
            forces.index_add_(0, self.global_atom_idx[:, 0], -self.forces)

            # For the second atom in the pair, add the bond force vector
            forces.index_add_(0, self.global_atom_idx[:, 1], +self.forces)
            return energy, forces
            
    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)


class RepulsionCGForce(Force):
    def __init__(self,
                 term_to_pair: torch.Tensor,
                 global_atom_idx: torch.Tensor,
                 term_to_system: torch.Tensor,
                 term_to_param_idx: torch.Tensor,
                 atom_types: torch.Tensor,
                 kwarg_params: dict[str, torch.Tensor],
                 scale=1.0):
        """
        Repulsion force using a coarse-grained repulsion model.

        Parameters:
        - term_to_pair (Tensor): Maps the relevant repulsion pair indices to the global pair list.
        - global_atom_idx (Tensor): Atom index pairs (N, 2).
        - term_to_system (Tensor): Maps global pair to system index.
        - atom_types (Tensor): Atom type for each atom (int tensor of shape (n_atoms,)).
        - kwarg_params (dict): Contains precomputed per-pair 'coef' tensor of shape (npairs,).
        - scale (float): Scaling factor for energy/force.
        """
        super().__init__("RepulsionCG")

        # Persistent buffers
        self.term_to_pair = nn.parameter.Buffer(term_to_pair, persistent=True)
        self.global_atom_idx = nn.parameter.Buffer(global_atom_idx, persistent=True)
        self.term_to_system = nn.parameter.Buffer(term_to_system, persistent=True)
        self.term_to_param_idx = nn.parameter.Buffer(term_to_param_idx, persistent=True) if term_to_param_idx is not None else None
        self.atom_types = nn.parameter.Buffer(atom_types, persistent=True)
        self.scale = scale

        if "coef" not in kwarg_params:
            raise ValueError(
                "RepulsionCGForce requires a per-pair 'coef' tensor in kwarg_params "
                "(see fiumd.ensembles.build_repulsion_coefs)."
            )
        if kwarg_params["coef"].shape != (self.term_to_pair.shape[0],):
            raise ValueError(
                f"'coef' must have shape ({self.term_to_pair.shape[0]},), "
                f"got {tuple(kwarg_params['coef'].shape)}"
            )
        self.coef = nn.parameter.Buffer(kwarg_params["coef"], persistent=True)  # (npairs,)

        # Temporary buffers (non-persistent) matching coef's device and dtype
        npairs = self.term_to_pair.shape[0]
        buf_device = self.coef.device
        buf_dtype = self.coef.dtype
        self.pot = nn.parameter.Buffer(torch.empty(npairs, device=buf_device, dtype=buf_dtype), persistent=False)
        self.forces = nn.parameter.Buffer(torch.empty((npairs, 3), device=buf_device, dtype=buf_dtype), persistent=False)
        self.rinv1 = nn.parameter.Buffer(torch.empty(npairs, device=buf_device, dtype=buf_dtype), persistent=False)
        self.rinv6 = nn.parameter.Buffer(torch.empty(npairs, device=buf_device, dtype=buf_dtype), persistent=False)

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
        kwarg_params = (
            dict(system.params.nonbonded_params)
            if system.params.nonbonded_params is not None
            else {}
        )

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
        with torch.inference_mode():
            self.rinv1.copy_(ensemble.pairs_distances[self.term_to_pair])  # (npairs,)
            self.rinv1.reciprocal_()                                       # 1/r
            self.rinv6.copy_(self.rinv1).pow_(6)                           # 1/r^6
    
            self.pot.copy_(self.coef).mul_(self.rinv6).div_(self.scale)   # V_CG = B * (1/r)^6 / scale
    
            # F = -6 * B / r^7 / scale = -6 * V_CG / r * unitvec
            self.forces.copy_(ensemble.pairs_unitvec[self.term_to_pair])
            self.forces.mul_(-6).mul_(self.pot.unsqueeze(-1)).mul_(self.rinv1.unsqueeze(-1))

        if inplace:
            ensemble.energy.index_add_(0, self.term_to_system, self.pot)
            ensemble.forces.index_add_(0, self.global_atom_idx[:, 0], -self.forces)
            ensemble.forces.index_add_(0, self.global_atom_idx[:, 1], +self.forces)
            return None
        else:
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.term_to_system, self.pot)
            forces.index_add_(0, self.global_atom_idx[:, 0], -self.forces)
            forces.index_add_(0, self.global_atom_idx[:, 1], +self.forces)
            return energy, forces

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)


class TorchMDNetForce(Force):
    def __init__(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        batch: torch.Tensor,
        dtype=torch.float32,
        device="cpu",
        box=None,
    ):
        super().__init__("TorchMDNet")
        self.dtype = dtype
        # Put model in eval mode and move to device and dtype
        self.model = model.eval().to(device=device, dtype=self.dtype)
        # Freeze model parameters so only positions require grad
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register buffers
        self.box = nn.parameter.Buffer(
            box.to(device=device, dtype=self.dtype)
        ) if box is not None else None
        self.embeddings = nn.parameter.Buffer(embeddings.to(device=device))
        self.batch = nn.parameter.Buffer(batch.to(device=device))

    def forward(self, ensemble, inplace=True):
        # Ensure positions require grad and match model dtype
        pos = ensemble.positions.detach().to(dtype=self.dtype).clone().requires_grad_(True)
     
        self.box = self.box.detach().clone() if self.box is not None else None
        self.batch = self.batch.detach().clone() if self.batch is not None else None
        self.embeddings = self.embeddings.detach().clone()

        # 1) Forward through model (normal grad tracking)
        energy, _ = self.model(
            self.embeddings,
            pos,
            self.batch,
            self.box,
        )

        # 2) Compute forces as negative gradient w.r.t. positions
        forces = -torch.autograd.grad(
            energy,
            pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=False,
            retain_graph=False,
        )[0]

        # 3) Accumulate into the ensemble
        if inplace:
            ensemble.energy.add_(energy.detach().squeeze().to(dtype=ensemble.energy.dtype))
            ensemble.forces.add_(forces.to(dtype=ensemble.forces.dtype))
            return None
        else:
            return (
                energy.to(dtype=ensemble.energy.dtype),
                forces.to(dtype=ensemble.forces.dtype),
            )


# class TorchMDNetForce(Force):
#     def __init__(self,
#                  model: nn.Module,
#                  embeddings: torch.Tensor,
#                  batch: torch.Tensor,
#                  # output_transform=None,
#                  dtype=torch.float32,
#                  device="cpu",
#                  box=None, 
#                 ):

#         super().__init__("TorchMDNet")
#         self.model = model.eval().to(device)
#         # self.device = device
#         self.dtype = dtype
#         # box = getattr(ensemble, "box", None)

#         self.box = nn.parameter.Buffer(box.to(self.dtype)) if box is not None else None

#         # self.output_transformer = (
#         #     lambda e, f: (e, f) if output_transform is None else output_transform
#         # )

#         self.embeddings = nn.parameter.Buffer(embeddings)
#         self.batch = nn.parameter.Buffer(batch)

#         # self.pot = nn.parameter.Buffer(torch.empty(npairs), persistent=False)
#         # self.forces = nn.parameter.Buffer(torch.empty((npairs, 3)), persistent=False)

#     def forward(self, ensemble, inplace=True):
#         # 1) Enable no-grad for model weights/buffers:
#         with torch.inference_mode():
#             # We still keep requires_grad on positions so we can take grad below
#             energy, _ = self.model(
#                 self.embeddings,
#                 ensemble.positions,
#                 self.batch,
#                 self.box,
#             )

#         # 2) Compute forces as negative gradient w.r.t. positions
#         #    NOTE: create_graph=False since we don't need higher-order grads
#         forces = -torch.autograd.grad(
#             energy,
#             ensemble.positions,
#             grad_outputs=torch.ones_like(energy),
#             create_graph=False,
#             retain_graph=False,
#         )[0]

#         # 3) Accumulate into the ensemble
#         if inplace:
#             # Add energy (we detach since we only need the value):
#             ensemble.energy.add_(energy.detach().squeeze())
#             # Add forces (already detached from model weights):
#             ensemble.forces.add_(forces)
#             return None
#         else:
#             return energy, forces

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)
        
class DihedralForce(Force):
    def __init__(self,
                 term_to_pair: torch.Tensor,
                 global_atom_idx: torch.Tensor,
                 term_to_system: torch.Tensor,
                 term_to_param_idx: torch.Tensor,
                 atom_types: torch.Tensor,
                 kwarg_params: dict[str, torch.Tensor],
                 ):
        super().__init__("Dihedral")

        self.term_to_pair = nn.parameter.Buffer(term_to_pair, persistent=True)
        self.global_atom_idx = nn.parameter.Buffer(global_atom_idx, persistent=True)
        self.term_to_system = nn.parameter.Buffer(term_to_system, persistent=True)
        self.term_to_param_idx = nn.parameter.Buffer(term_to_param_idx, persistent=True)

        self.k = nn.parameter.Buffer(kwarg_params['params'][:, 0], persistent=True)
        self.phi0 = nn.parameter.Buffer(kwarg_params['params'][:, 1], persistent=True)
        self.periodicity = nn.parameter.Buffer(kwarg_params['params'][:, 2], persistent=True)

        # Temporary buffers (non-persistent) matching parameter device and dtype
        nterms = self.term_to_pair.shape[0]
        buf_device = self.k.device
        buf_dtype = self.k.dtype
        self.pot = nn.parameter.Buffer(torch.empty(nterms, device=buf_device, dtype=buf_dtype), persistent=False)
        self.forces = nn.parameter.Buffer(torch.empty((nterms, 4, 3), device=buf_device, dtype=buf_dtype), persistent=False)


    @classmethod
    def build_params(cls, system):
        return system.params.dihedral_params

    def forward(self, ensemble, inplace=True):
        with torch.inference_mode():
            evaluate_torsion_into(
                self.pot,
                self.forces,
                all_relative_positions=ensemble.pairs_relative_positions,
                all_distances=ensemble.pairs_distances,
                dihedral_idx_map=self.term_to_pair,
                torsion_param_map=self.term_to_param_idx,
                k=self.k,
                phi0=self.phi0,
                periodicity=self.periodicity,
            )
    
        if inplace:
            ensemble.energy.index_add_(0, self.term_to_system, self.pot)
            ensemble.forces.index_add_(0,
                                       self.global_atom_idx.view(-1),
                                       self.forces.view(-1, 3)
                                       )
            return None
        else:
            energy = torch.zeros_like(ensemble.energy)
            forces = torch.zeros_like(ensemble.forces)
            energy.index_add_(0, self.term_to_system, self.pot)
            forces.index_add_(0,
                              self.global_atom_idx.view(-1),
                              self.forces.view(-1, 3)
                              )
            return energy, forces

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)

# def requires_grad_positions(force_fn: Force):
#     def wrapper(ensemble, *args, **kwargs):
#         # 1. Turn on gradient tracking
#         ensemble.positions.requires_grad_(True)
        
#         # 2. Call the force function
#         result = force_fn(ensemble, *args, **kwargs)

#         # 3. Detach to allow in-place MD integration
#         ensemble.positions = ensemble.positions.detach()

#         return result
#     return wrapper
        
def evaluate_torsion_into(
    pot: torch.Tensor,
    forces: torch.Tensor,  # shape (n_terms, 4, 3)
    *,
    all_relative_positions: torch.Tensor,
    all_distances: torch.Tensor,
    dihedral_idx_map: torch.Tensor,     # shape (n_terms, 3)
    torsion_param_map: torch.Tensor,    # shape (n_terms, 1)
    k: torch.Tensor,
    phi0: torch.Tensor,
    periodicity: torch.Tensor,
):
    """
    Computes torsion energy and forces for a set of dihedral terms.
    """
    eps = 1e-8
    dih_idx = torsion_param_map#[:, 0]

    r12 = all_relative_positions[dihedral_idx_map[dih_idx, 0]]  # shape (n_terms, 3)
    r23 = all_relative_positions[dihedral_idx_map[dih_idx, 1]]
    r34 = all_relative_positions[dihedral_idx_map[dih_idx, 2]]

    crossA = torch.cross(r12, r23, dim=1)
    crossB = torch.cross(r23, r34, dim=1)
    crossC = torch.cross(r23, crossA, dim=1)

    normA = crossA.norm(dim=1).clamp(min=eps)
    normB = crossB.norm(dim=1).clamp(min=eps)
    normC = crossC.norm(dim=1).clamp(min=eps)

    crossB_normed = crossB / normB.unsqueeze(1)

    cos_phi = (crossA * crossB_normed).sum(dim=1) / normA
    sin_phi = (crossC * crossB_normed).sum(dim=1) / normC
    phi = -torch.atan2(sin_phi, cos_phi)

    is_amber = (periodicity > 0).all()
    angle_diff = torch.where(
        is_amber,
        periodicity * phi - phi0,
        ((phi - phi0 + torch.pi) % (2 * torch.pi)) - torch.pi
    )

    pot.copy_(
        torch.where(
            is_amber,
            k * (1 + torch.cos(angle_diff)),
            k * angle_diff**2
        )
    )

    coeff = torch.where(
        is_amber,
        -periodicity * k * torch.sin(angle_diff),
        2 * k * angle_diff
    )

    norm_r23 = all_distances[dihedral_idx_map[dih_idx, 1]]
    norm_r23_sq = norm_r23.pow(2).clamp(min=eps)

    dot12 = (r12 * r23).sum(dim=1)
    dot34 = (r34 * r23).sum(dim=1)

    force_factor0 = (-coeff * norm_r23) / normA.pow(2)
    force_factor1 = dot12 / norm_r23_sq
    force_factor2 = dot34 / norm_r23_sq
    force_factor3 = (coeff * norm_r23) / normB.pow(2)

    force0vec = force_factor0.unsqueeze(1) * crossA
    force3vec = force_factor3.unsqueeze(1) * crossB
    s = force_factor1.unsqueeze(1) * force0vec - force_factor2.unsqueeze(1) * force3vec

    forces[:, 0] = -force0vec
    forces[:, 1] = force0vec + s
    forces[:, 2] = force3vec - s
    forces[:, 3] = -force3vec
