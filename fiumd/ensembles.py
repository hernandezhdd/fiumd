import torch
import torch.nn as nn
# from fiumd.forces import Force
import fiumd.forces
import fiumd.systems
import fiumd.utils
import numpy as np
from typing import Type
import inspect

# from fiumd.neighbours import NeighbourListBuilder
import fiumd.neighbours

from torchmdnet.models.model import load_model

class Ensemble(nn.Module):    
    """
    Batched molecular dynamics ensemble for simulating multiple molecular systems in parallel.

    The Ensemble class aggregates atomic data, topologies, and force field parameters from multiple 
    `System` instances and enables efficient batched computations of forces and energies. It handles 
    device and dtype consistency, tracks global and per-system atom indexing, and supports various 
    force terms (e.g., bonds, repulsions) via a modular force registry.

    Features:
    ---------
    - Parallel simulation of multiple systems with support for variable atom counts.
    - Efficient global indexing for per-system bonded/nonbonded interactions.
    - Force term construction via force class registry and parameter dictionaries.
    - Device and dtype-aware tensor management.
    - In-place updates for performance-critical geometry and force operations.
    - Utility methods for data introspection, geometry reduction, and system-level grouping.

    Parameters:
    -----------
    systems : List[System]
        List of `System` instances to include in the ensemble.
    device : str or torch.device, optional
        Device to allocate all tensors on (default: "cpu").
    dtype : torch.dtype, optional
        Floating-point precision for numerical tensors (default: torch.float32).

    Attributes:
    -----------
    systems : List[System]
        The list of individual molecular systems.
    natoms_per_system : torch.Tensor
        Number of atoms in each system.
    total_atoms : int
        Total number of atoms across all systems.
    atom_offsets : torch.Tensor
        Cumulative atom offset for indexing global tensors by system.
    atom_to_system : torch.Tensor
        Map from atom index to its system index.
    atom_types : List[str]
        List of atom type names across all systems.
    atom_type_map : dict
        Mapping from atom type string to unique integer index.
    atom_type_indices : torch.Tensor
        Per-atom tensor of type indices for embedding/lookup.
    positions, velocities, forces : torch.Tensor
        Tensor of shape (total_atoms, 3) storing simulation data.
    energy : torch.Tensor
        Per-system potential energy values.
    box : torch.Tensor
        Per-system simulation box dimensions, shape (nsystems, 3).
    global_pair_idx : torch.Tensor
        Tensor of shape (n_pairs, 2) indexing atom pairs globally.
    pair_to_system : torch.Tensor
        Map from each global pair to its parent system index.
    force_modules : nn.ModuleList
        Registered force modules, compatible with PyTorch.
    pairs_relative_positions, pairs_distances, pairs_unitvec : torch.Tensor
        Buffers for pairwise geometry used in force calculations.
    
    Methods:
    --------
    set_positions(tensor)
        Set new positions while enforcing device/dtype.
    set_velocities(tensor)
        Set new velocities while enforcing device/dtype.
    set_box(tensor)
        Set new box dimensions per system.
    centers_of_mass()
        Compute center of mass for each system.
    centers_of_geom()
        Compute center of geometry for each system.
    split_by_system(tensor)
        Split a global tensor (shape: [total_atoms, ...]) into per-system chunks.
    reduce_by_system(tensor, reduce_fn, **kwargs)
        Reduce a global tensor per system (e.g., torch.mean over atoms).
    compute_forces(inplace=True)
        Compute all force contributions and update energy/forces.
    info()
        Print internal attributes (shapes, dtypes, devices) for quick inspection.

    Example:
    --------
    >>> ensemble = Ensemble(systems=my_systems, parameters=my_params, device="cuda", dtype=torch.float64)
    >>> ensemble.set_positions(new_positions)
    >>> ensemble.compute_forces()
    >>> print(ensemble.energy)
    >>> ensemble.info()
    """
    def __init__(self, systems: list, device="cpu",
                 dtype=torch.float32, cutoff=5.0,
                nnp_model=None, nnp_embeddings=None):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.systems = systems
        self.nsystems = len(systems)
        self.nnp_model = nnp_model
        self.nnp_embeddings = nnp_embeddings
        
        self.natoms_per_system = torch.tensor([s.natoms for s in systems], device=self.device, dtype=torch.long)
        self.total_atoms = self.natoms_per_system.sum()

        self.atom_offsets = torch.tensor([0] + [s.natoms for s in systems][:-1], device=self.device, dtype=torch.long).cumsum(0)
        self.atom_to_system = torch.repeat_interleave(torch.arange(self.nsystems, device=self.device, dtype=torch.long), self.natoms_per_system)

        self.atom_types = sum([s.params.atomtypes.tolist() for s in systems], [])
        self.atom_type_map = {t: i for i, t in enumerate(sorted(set(self.atom_types)))}
        self.atom_type_indices = torch.tensor([self.atom_type_map[t] for t in self.atom_types], device=self.device, dtype=torch.long)

        self.masses = torch.cat([s.params.masses.to(self.device, self.dtype) for s in systems], dim=0)
        self.inv_masses = 1.0 / self.masses

        self.positions = torch.zeros((self.total_atoms, 3), device=self.device, dtype=self.dtype)
        self.velocities = torch.zeros_like(self.positions)
        self.forces = torch.zeros_like(self.positions)
        self.energy = torch.zeros(self.nsystems, device=self.device, dtype=self.dtype)

        self.box = torch.stack([s.box.to(self.device, self.dtype) for s in systems], dim=0)

        self._build_global_pairs_and_maps()

        self._box_expanded = self.box[self.pair_to_system]  # shape: (n_pairs, 3)
        self.n_pairs = self.global_pair_idx.shape[0]

        self.pairs_distances = torch.empty((self.n_pairs,), device=self.device, dtype=self.dtype)
        self.pairs_unitvec = torch.empty((self.n_pairs, 3), device=self.device, dtype=self.dtype)
        self.pairs_relative_positions = torch.empty((self.n_pairs, 3), device=self.device, dtype=self.dtype)
        
        self._pbc_offset = torch.empty((self.n_pairs, 3), device=self.device, dtype=self.dtype)
        

        # Assume build_dihedral_params returns something like (n_dihedral_terms, param_dim)
        # self.dihedral_params = build_dihedral_params(...)
        # self.n_triplets = None
        # self.n_dihedrals = self.global_dihedral_indices.shape[0]
        # self.n_dihedral_terms = self.dihedral_params.shape[0]
        self.nl_builder = fiumd.neighbours.NeighbourListBuilder(n_pairs=self.n_pairs, cutoff=cutoff, device=self.device, dtype=self.dtype)

        # self.triplet_positions = torch.empty((self.n_triplets, 3, 3),
        #                                      device=self.device,
        #                                      dtype=self.dtype)
        self.force_modules = nn.ModuleList(self._build_force_terms())

        # self.quadruplet_positions = torch.empty((n_dihedrals, 3, 4),
        #                                         device=self.device,
        #                                         dtype=self.dtype)
    
    @property
    def centers_of_geom(self):
        self.centers_of_geom_ = self.reduce_by_system(self.positions, torch.mean, dim=0)  # atomwise mean xyz
        return self.centers_of_geom_

    @property
    def centers_of_mass(self):
        total_system_inv_mass = 1.0 / self.reduce_by_system(self.masses, torch.sum, dim=0).unsqueeze(-1)
        self.centers_of_mass_ = total_system_inv_mass * self.reduce_by_system(self.positions * self.masses.unsqueeze(-1), torch.sum, dim=0)  # atomwise mean xyz
        return self.centers_of_mass_

    def _build_global_pairs_and_maps(self):
        device = self.device

        self._force_terms_set = set()
        self.term_to_pair_by_force = {}
        self.global_atom_idx_by_force = {}
        self.term_to_system_by_force = {}
        self.term_to_param_idx_by_force = {}
        self._param_names_by_force = {}

        for s in self.systems:
            self._force_terms_set.update(s.params_by_force.keys())
            for force_name, params in s.params_by_force.items():
                if force_name not in self._param_names_by_force:
                    self._param_names_by_force[force_name] = params.keys()

        # Build all (i < j) intra-system pairs
        all_pairs_by_system = []
        for s in self.systems:
            idx = torch.arange(s.natoms)
            i, j = torch.meshgrid(idx, idx, indexing='ij')
            mask = i < j
            all_pairs = torch.stack([i[mask], j[mask]], dim=1)
            all_pairs_by_system.append(all_pairs)

        self.global_pair_idx, self.pair_to_system = local_to_global_idxs(
            all_pairs_by_system, self.natoms_per_system, device=device
        )
        all_pairs_set = set( torch.arange(self.global_pair_idx.shape[0]).tolist() )

        for force_name in self._force_terms_set:
            if force_name == "RepulsionCGForce":
                continue

            # local_idxs = [s.params_by_force[force_name]['idx'] for s in self.systems]
            # local_maps = [s.params_by_force[force_name].get('map') for s in self.systems]

            local_maps = {s: s.params_by_force[force_name]['map'] for s in self.systems}
            local_idxs = [
                s.params_by_force[force_name]['idx'][local_maps[s][:, 0]]
                for s in self.systems
            ]
            local_param_idxs = [local_maps[s][:, 1] for s in self.systems]

            global_atom_idx, term_to_system = local_to_global_idxs(
                local_idxs, self.natoms_per_system, device=device
            )
                  
            if force_name == "DihedralForce":
                # map dihedral pairs to global pairs indices (three edges per dihedral)
                dihedral_pair_maps = torch.stack([
                    match_pair_indices(self.global_pair_idx,
                                       global_atom_idx[:, [i, i + 1]],
                                       max_atom=self.total_atoms)
                    for i in range(3)
                ], dim=1)  # shape (n_dihedrals, 3)

                nterms_per_system = torch.tensor([local_idx.shape[0] for local_idx in local_idxs],
                                                  device=self.device)

                term_to_param_idx, param_to_system = local_to_global_idxs(
                                                    local_param_idxs, nterms_per_system, device=device
                                                )

                # combine per-system param maps into single tensor
                # term_to_param_idx = torch.cat([m[:, 1:2] for m in local_maps], dim=0).squeeze(1)

                # expand dihedral maps to per-term maps
                # term_to_pair = dihedral_pair_maps[term_to_param_idx]           # (n_terms,3)
                # term_global_atom_idx = global_atom_idx[term_to_param_idx]      # (n_terms,4)
                # term_to_system_expanded = term_to_system[term_to_param_idx]     # (n_terms,)

                self.term_to_pair_by_force[force_name] = dihedral_pair_maps
                self.global_atom_idx_by_force[force_name] = global_atom_idx
                self.term_to_system_by_force[force_name] = term_to_system
                self.term_to_param_idx_by_force[force_name] = term_to_param_idx
                # print(term_to_param_idx.shape)

            else:
                # 2-body forces: direct mapping
                term_to_pair = match_pair_indices(
                    self.global_pair_idx, global_atom_idx, self.total_atoms
                )  # (n_terms,)

                self.term_to_pair_by_force[force_name] = term_to_pair
                self.global_atom_idx_by_force[force_name] = global_atom_idx
                self.term_to_system_by_force[force_name] = term_to_system
                self.term_to_param_idx_by_force[force_name] = None



        # RepulsionCGForce = all nonbonded intra-system pairs
        bond_pairs_set = set( self.term_to_pair_by_force["BondForce"].tolist() )
        repulsion_pairs_set = all_pairs_set - bond_pairs_set
        repulsion_pair_idx = torch.tensor(sorted(list(repulsion_pairs_set)), dtype=torch.long, device=device)

        # self.term_to_pair_by_force["RepulsionCGForce"] = match_pair_indices(
        #     self.global_pair_idx, repulsion_pair_idx, max_atom=self.total_atoms
        # )
        self.term_to_pair_by_force["RepulsionCGForce"] = repulsion_pair_idx
        self.global_atom_idx_by_force["RepulsionCGForce"] = self.global_pair_idx[repulsion_pair_idx]
        self.term_to_system_by_force["RepulsionCGForce"] = self.pair_to_system[repulsion_pair_idx]
        self.term_to_param_idx_by_force["RepulsionCGForce"] = None

    def _build_force_terms(self):
        force_modules_list = []

        for force_name in self._force_terms_set:
            force_cls = fiumd.forces.Force.registry[force_name]

            kwarg_params = {}
            for key in self._param_names_by_force[force_name]:
                if key == "B":
                    B = merge_B_matrices(self.systems, self.atom_type_map, device=self.device)
                    kwarg_params["B"] = B.to(self.device)
                else:
                    key_list = [s.params_by_force[force_name][key] for s in self.systems]
                    kwarg_params[key] = torch.cat(key_list, dim=0).to(self.device)

            force_module = force_cls(
                term_to_pair=self.term_to_pair_by_force[force_name],
                global_atom_idx=self.global_atom_idx_by_force[force_name],
                term_to_system=self.term_to_system_by_force[force_name],
                term_to_param_idx=self.term_to_param_idx_by_force.get(force_name),
                atom_types=self.atom_type_indices,
                kwarg_params=kwarg_params,
            )

            force_module.eval()

            force_modules_list.append(force_module)

        if self.nnp_model:
            force_module = fiumd.forces.TorchMDNetForce(
                model=self.nnp_model,
                embeddings=self.nnp_embeddings,
                batch=self.atom_to_system,
                device=self.device,
                box=self.box[0].unsqueeze(1).expand(3, 3) * torch.eye(3, device=self.device)
            )
            force_modules_list.append(force_module)

        return force_modules_list



    def _update_pairs_geometry_(self):
        """
        Updates pairwise distances, directions, and relative vectors
        for all defined interactions, in-place.
    
        This method:

            - Computes relative vectors with PBC: `self.pairs_relative_positions`
            - Computes pairwise distances: `self.pairs_distances`
            - Computes unit direction vectors: `self.pairs_unitvec`
        """
    
        # Compute relative vectors
        self.pairs_relative_positions.copy_(self.positions[self.global_pair_idx[:, 0]])
        self.pairs_relative_positions.sub_(self.positions[self.global_pair_idx[:, 1]])
    
        # Apply PBC — compute box offset into a buffer first
        self._pbc_offset.copy_(self.pairs_relative_positions)
        self._pbc_offset.div_(self._box_expanded)
        self._pbc_offset.round_()
        self._pbc_offset.mul_(self._box_expanded)
        self.pairs_relative_positions.sub_(self._pbc_offset)
    
        # Compute distances
        self.pairs_distances.copy_(self.pairs_relative_positions.norm(dim=-1))
    
        # Compute unit vectors (re-using pairs_relative_positions buffer to avoid temp)
        self.pairs_unitvec.copy_(self.pairs_relative_positions)
        self.pairs_unitvec.div_(self.pairs_distances.unsqueeze(-1).clamp_min(1e-8))  # Avoid division by 0


    def compute_forces(self, inplace=True):
        """
        Compute forces and potential energy for all force terms.
    
        Parameters
        ----------
        inplace : bool
            If True, updates ensemble.force and ensemble.energy in-place.
            If False, returns a dict of results for each force term.
    
        Returns
        -------
        None if inplace=True,
        else dict of {force_name: (energy, force)} for each term.
        """
        self.zero_energy_()
        self.zero_forces_()

        if inplace:
            # TorchMDNetForce: must compute with autograd enabled
            for f in self.force_modules:
                if isinstance(f, fiumd.forces.TorchMDNetForce):
                    f(self, inplace=True)

            # All other forces: can run inside inference_mode
            with torch.inference_mode():
                for f in self.force_modules:
                    if not isinstance(f, fiumd.forces.TorchMDNetForce):
                        f(self, inplace=True)
            return None
        else:
            results = {}
            for f in self.force_modules:
                if isinstance(f, fiumd.forces.TorchMDNetForce):
                    pot, force_tensor = f(self, inplace=False)
                    results[f.name] = (pot, force_tensor)
      
            with torch.inference_mode():
                for f in self.force_modules:
                    if not isinstance(f, fiumd.forces.TorchMDNetForce):
                        pot, force_tensor = f(self, inplace=False)
                        results[f.name] = (pot, force_tensor)
                        
            return results
        
    def set_positions(self, positions: torch.Tensor):
        assert positions.shape == self.positions.shape, \
            f"Expected shape {self.positions.shape}, got {positions.shape}"
        self.positions = positions.detach().to(self.device, self.dtype)
        # self.positions.requires_grad_(False)

        self._update_pairs_geometry_()
        self.compute_forces()  # Initialize forces consistent with the new positions
        self.forces.requires_grad_(False)
        
    def set_velocities(self, velocities: torch.Tensor):
        assert velocities.shape == self.velocities.shape, \
            f"Expected shape {self.velocities.shape}, got {velocities.shape}"
        self.velocities = velocities.to(self.device, self.dtype)
        self.velocities.requires_grad_(False)


    def set_box(self, box: torch.Tensor):
        """
        Update the simulation box for each system. Will also update _box_expanded.
        """
        assert box.shape == self.box.shape, f"Expected shape {self.box.shape}, got {box.shape}"
        self.box.copy_(box.to(self.device, self.dtype))
    
        # Update expanded per-pair box values
        self._box_expanded = self.box[self.pair_to_system]  # re-create on demand

    def _update_positions_(self, new_positions: torch.Tensor):
        """
        In-place update of atom positions.
    
        Parameters
        ----------
        new_positions : torch.Tensor
            Shape: (n_all_atoms, 3)
            The updated atom positions (e.g., after integration).
        """
        self.positions.copy_(new_positions)

    def _update_velocities_(self, new_velocities: torch.Tensor):
        """
        In-place update of atom velocities.
    
        Parameters
        ----------
        new_velocities : torch.Tensor
            Shape: (n_all_atoms, 3)
        """
        self.velocities.copy_(new_velocities)
    
    
    def _update_forces_(self, new_forces: torch.Tensor):
        """
        In-place update of atom forces.
    
        Parameters
        ----------
        new_forces : torch.Tensor
            Shape: (n_all_atoms, 3)
        """
        self.forces.copy_(new_forces)

    def zero_forces_(self):
        """
        Zeros the force tensor in-place.
        """
        self.forces.zero_()

    def zero_energy_(self):
        """
        Zeros the force tensor in-place.
        """
        self.energy.zero_()

    def split_by_system(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Splits a global tensor (e.g., positions, velocities) by system.
    
        Args:
            tensor (torch.Tensor): A tensor with shape (total_atoms, ...) where the first dimension
                                   corresponds to all atoms stacked across systems.
    
        Returns:
            List[torch.Tensor]: A list of per-system tensors.
        """
    
        split_tensors = [tensor[offset:offset + natoms] 
                         for offset, natoms in zip(self.atom_offsets, self.natoms_per_system)]
    
        return split_tensors

    def reduce_by_system(self, tensor: torch.Tensor, reduce_fn: callable, **kwargs) -> torch.Tensor:
        """
        Applies a reduction function (e.g., torch.mean, torch.sum) per system.
    
        Args:
            tensor (torch.Tensor): Tensor of shape (total_atoms, ...), stacked across systems.
            reduce_fn (callable): Function like torch.mean, torch.sum, or a custom reduction.
            **kwargs: Extra keyword arguments passed to reduce_fn (e.g., dim=1).
    
        Returns:
            torch.Tensor: Shape (nsystems, ...) with reduced values per system.
        """
        result = [
            reduce_fn(tensor[offset:offset + natoms], **kwargs)
            for offset, natoms in zip(self.atom_offsets, self.natoms_per_system)
        ]
        return torch.stack(result, dim=0)
        

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)
    



def update_pairs_geometry_(pairs_positions, box, pair_idx,
                           out_distances, out_unitvec, out_relative_positions):
    """
    Compute geometry (distance, unit vectors, relative vectors) between atom pairs.
    All outputs are written in-place.
    """
    src = pairs_positions[:, :, 0]  # (n_pairs, 3)
    dst = pairs_positions[:, :, 1]  # (n_pairs, 3)

    delta = src - dst

    # Handle periodicity
    if box.ndim == 1:
        box_expanded = box.unsqueeze(0)  # (1, 3)
    elif pair_idx is not None:
        box_expanded = box[pair_idx]  # (n_pairs, 3)
    else:
        raise ValueError("pair_idx must be provided if box is per-system")

    delta -= box_expanded * torch.round(delta / box_expanded)

    dist = torch.norm(delta, dim=-1)
    unitvec = delta / (dist.unsqueeze(-1) + 1e-8)  # Safe division

    out_distances.copy_(dist)
    out_unitvec.copy_(unitvec)
    out_relative_positions.copy_(delta)

    
def calc_dists(atom_pos, box, pair_idx=None):
    """
    Parameters
    ----------
    atom_pos : torch.Tensor
        Shape: (n_pairs, 3, 2) where [..., 0] is atom1, [..., 1] is atom2.
    box : torch.Tensor
        Shape: (3,) or (nsystems, 3)
    pair_idx : torch.Tensor or None
        Shape: (n_pairs,) mapping each pair to its system if box is per-system.

    Returns
    -------
    dist : torch.Tensor
        Shape: (n_pairs,)
    direction_unitvec : torch.Tensor
        Shape: (n_pairs, 3)
    direction_vec : torch.Tensor
        Shape: (n_pairs, 3)
    """
    src = atom_pos[:, :, 0]  # (n_pairs, 3)
    dst = atom_pos[:, :, 1]  # (n_pairs, 3)

    # Compute vector from dst to src
    delta = src - dst  # (n_pairs, 3)

    # Apply periodic boundary conditions
    if box.ndim == 1:
        box_expanded = box.unsqueeze(0)  # (1, 3)
    elif pair_idx is not None:
        box_expanded = box[pair_idx]  # (n_pairs, 3)
    else:
        raise ValueError("pair_idx must be provided if box is per-system (nsystems, 3)")

    # Wrap distances under PBC
    delta = delta - box_expanded * torch.round(delta / box_expanded)

    dist = torch.norm(delta, dim=-1)  # (n_pairs,)
    direction_unitvec = delta / (dist.unsqueeze(-1) + 1e-8)  # Avoid div-by-zero
    return dist, direction_unitvec, delta


def match_pair_indices(unique_pairs: torch.Tensor,
                       target_pairs: torch.Tensor,
                       max_atom: int):
    """
    unique_pairs: (n_unique, 2)
    target_pairs: (n_target, 2)
    Returns: index_map of shape (n_target,)
    """
    # Make sure both are sorted so (i,j) == (j,i)
    unique_pairs = torch.sort(unique_pairs, dim=-1).values
    target_pairs = torch.sort(target_pairs, dim=-1).values

    # Compute hashes
    unique_hashes = unique_pairs[:, 0] * max_atom + unique_pairs[:, 1]
    target_hashes = target_pairs[:, 0] * max_atom + target_pairs[:, 1]
    # Sort unique hashes
    sorted_hashes, sorted_indices = torch.sort(unique_hashes)
    
    # Map target to sorted unique hashes
    index_in_sorted = torch.searchsorted(sorted_hashes, target_hashes)

    # Use sorted_indices to get original positions
    index_map = sorted_indices[index_in_sorted]
    return index_map

def local_to_global_idxs(local_pair_lists, natoms_per_system, device="cpu"):
    """
    Convert local atom pair indices (per system) to global indices.
    
    Parameters
    ----------
    local_pair_lists : list of torch.Tensor
        Each tensor has shape (n_pairs_in_system, 2), with local atom indices.
    natoms_per_system : list or torch.Tensor
        Number of atoms in each system, len = nsystems.

    Returns
    -------
    global_pair_idx : torch.Tensor
        Shape (total_pairs, 2), global atom indices.
    pair_to_system : torch.Tensor
        Shape (total_pairs,), maps each pair to its system index.
    """
    atom_offsets = torch.cumsum(torch.cat([natoms_per_system.new_zeros(1), natoms_per_system[:-1]]), dim=0).to(device)

    global_pair_idx = []
    pair_to_system = []

    for system_idx, (offset, local_pairs) in enumerate(zip(atom_offsets, local_pair_lists)):
        global_pairs = local_pairs.to(device) + offset  # broadcast add
        global_pair_idx.append(global_pairs)
        pair_to_system.append(torch.full((local_pairs.shape[0],), system_idx, dtype=torch.long, device=device))

    return torch.cat(global_pair_idx, dim=0), torch.cat(pair_to_system, dim=0)


def merge_B_matrices(systems, global_type_map, device):
    """
    systems: list of system objects
    global_type_map: dict from atomtype string to global index
    """
    ntypes = len(global_type_map)
    B_global = torch.zeros((ntypes, ntypes), dtype=torch.float32, device=device)
    count = torch.zeros((ntypes, ntypes), dtype=torch.float32, device=device)  # to allow averaging

    for system in systems:
        local_types = system.params.atomtypes.tolist()
        local_type_to_idx = {t: i for i, t in enumerate(sorted(set(local_types)))}
        B_local = system.params_by_force["RepulsionCGForce"]["B"]  # shape (n_local_types, n_local_types)

        for t1 in local_types:
            for t2 in local_types:
                i_local = local_type_to_idx[t1]
                j_local = local_type_to_idx[t2]
                i_global = global_type_map[t1]
                j_global = global_type_map[t2]

                B_global[i_global, j_global] += B_local[i_local, j_local]
                count[i_global, j_global] += 1

    # Average where multiple systems contributed
    mask = count > 0
    B_global[mask] /= count[mask]

    return B_global