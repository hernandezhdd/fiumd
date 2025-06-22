# from scipy import constants as const
import torch
import numpy as np
from math import pi


class Forces:
    """
    Parameters (Version modificada)
    ----------
    cutoff : float
        If set to a value it will only calculate LJ, electrostatics and bond energies for atoms which are closer
        than the threshold
    rfa : bool
        Use with `cutoff` to enable the reaction field approximation for scaling of the electrostatics up to the cutoff.
        Uses the value of `solventDielectric` to model everything beyond the cutoff distance as solvent with uniform
        dielectric.
    solventDielectric : float
        Used together with `cutoff` and `rfa`
    """

    # 1-4 is nonbonded but we put it currently in bonded to not calculate all distances
    bonded = ["bonds", "angles", "dihedrals", "impropers", "1-4"]
    nonbonded = ["electrostatics", "lj", "repulsion", "repulsioncg"]
    terms = bonded + nonbonded

    def __init__(
        self,
        parameters,
        terms=None,
        external=None,
        cutoff=None,
        rfa=False,
        solventDielectric=78.5,
        switch_dist=None,
        exclusions=("bonds", "angles", "1-4"),
    ):
        self.par = parameters
        if terms is None:
            raise RuntimeError(
                'Set force terms or leave empty brackets [].\nAvailable options: "bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj", "repulsion", "repulsioncg".'
            )

        # Precalculate A, B arrays for nonbonded terms
#        if self.par.nonbonded_params is not None and "lj" in terms:
        if self.par.nonbonded_params is not None:
            self.par.A, self.par.B = self.par.get_AB()

        # if self.par.nonbonded_14_params is not None and "lj" in terms:
        #     self.par.A14, self.par.B14 = self.par.get_AB_14()

        self.energies = [ene.lower() for ene in terms]
        for et in self.energies:
            if et not in Forces.terms:
                raise ValueError(f"Force term {et} is not implemented.")

        if "1-4" in self.energies and "dihedrals" not in self.energies:
            raise RuntimeError(
                "You cannot enable 1-4 interactions without enabling dihedrals"
            )

        self.natoms = len(parameters.masses)
        self.require_distances = any(f in self.nonbonded for f in self.energies)
        self.ava_idx = (
            self._make_indeces(
                self.natoms, parameters.get_exclusions(exclusions), parameters.device
            )
            if self.require_distances
            else None
        )

        # print(f"self.ava_idx.shape: {self.ava_idx.shape}")
        
        self.external = external
        self.cutoff = cutoff
        self.rfa = rfa
        self.solventDielectric = solventDielectric
        self.switch_dist = switch_dist

    def _filter_by_cutoff(self, dist, arrays):
        under_cutoff = dist <= self.cutoff
        indexedarrays = []
        for arr in arrays:
            indexedarrays.append(arr[under_cutoff])
        return indexedarrays

    def compute(
        self,
        pos,
        box,
        forces,
        returnDetails=False,
        explicit_forces=True,
        toNumpy=True,
        calculateForces=True,
        return_forces=False,
    ):

        if calculateForces:
            if not explicit_forces and not pos.requires_grad:
                raise RuntimeError(
                    "The positions passed don't require gradients. Please use pos.detach().requires_grad_(True) before passing."
                )
        else:
            explicit_forces = False

        nsystems = pos.shape[0]

        # pot = []
        # for i in range(nsystems):
        #     pp = {
        #         v: torch.zeros(1, device=pos.device).type(pos.dtype)
        #         for v in self.energies
        #     }
        #     pp["external"] = torch.zeros(1, device=pos.device).type(pos.dtype)
        #     pot.append(pp)

        # Initialize potential energy accumulator
        pot = {
            v: torch.zeros(nsystems, device=pos.device, dtype=pos.dtype)
            for v in self.energies
        }
        pot["external"] = torch.zeros(nsystems, device=pos.device, dtype=pos.dtype)

        if forces is not None:
            forces.zero_()
            #forces = torch.zeros_like(forces)

            
        spos = pos
        mask = torch.eye(3, device=box.device).bool()
        sbox = box[0][mask].unsqueeze(0).expand(nsystems, -1)

        # Bonded terms
            # TODO: We are for sure doing duplicate distance calculations here!
        if "bonds" in self.energies and self.par.bond_params is not None:
            pairs = self.par.bond_params["idx"]
            pairs = pairs.unsqueeze(0).expand(nsystems, -1, -1)
            param_idx = self.par.bond_params["map"][:, 1]
            param_idx = param_idx.unsqueeze(0).expand(nsystems, -1)  # shape: (nsystems, nbonds)
    
            bond_dist, bond_unitvec, _ = calculate_distances(spos, pairs, sbox)
    
            bond_params = self.par.bond_params["params"][param_idx]
    #             bond_params = bond_params0.unsqueeze(0).expand(nsystems, -1, -1)
    
            # if self.cutoff is not None:
            #     (
            #         bond_dist,
            #         bond_unitvec,
            #         pairs,
            #         bond_params,
            #     ) = self._filter_by_cutoff(
            #         bond_dist, (bond_dist, bond_unitvec, pairs, bond_params)
            #     )
            E, force_coeff = evaluate_bonds(bond_dist, bond_params, explicit_forces)
    
            pot["bonds"] += E.sum(dim=1)  # E is (nsystems, nbonds) → sum over bonds
            
            # for i in range(nsystems):
                # pot[i]["bonds"] = pot[i]["bonds"] + E[i].sum()
                
            # if explicit_forces:
            forcevec = bond_unitvec * force_coeff.unsqueeze(-1)  # (nsystems, nbonds, 3)
        
            # Flatten batch for index_add_
            forces_flat = forces.view(-1, 3)  # (nsystems * natoms, 3)
            
            # Compute flat indices
            offset = torch.arange(nsystems, device=pos.device) * pos.shape[1]  # (nsystems,)
            flat_idx0 = (pairs[:, :, 0] + offset[:, None]).reshape(-1)  # (nsystems * nbonds)
            flat_idx1 = (pairs[:, :, 1] + offset[:, None]).reshape(-1)
            forcevec_flat = forcevec.reshape(-1, 3)  # (nsystems * nbonds, 3)
            
            # Apply index_add in parallel
            forces_flat.index_add_(0, flat_idx0, -forcevec_flat)
            forces_flat.index_add_(0, flat_idx1,  forcevec_flat)
            
            # for i in range(nsystems):
            #     # pot[i]["bonds"] = pot[i]["bonds"] + E[i].sum()
            #     forces[i].index_add_(0, pairs[i, :, 0], -forcevec[i])
            #     forces[i].index_add_(0, pairs[i, :, 1],  forcevec[i])

        if "dihedrals" in self.energies and self.par.dihedral_params is not None:
            dihed_idx = self.par.dihedral_params["idx"]
            dihed_idx = dihed_idx.unsqueeze(0).expand(nsystems, -1, -1) # dihed_idx: (nsystems, ntorsions, 4)
            param_idx = self.par.dihedral_params["map"][:, 1]
            param_idx = param_idx.unsqueeze(0).expand(nsystems, -1)
    
            _, _, r12 = calculate_distances(spos, dihed_idx[:, :, [0, 1]], sbox) # shape: (nsystems, ntorsions, 3)
            _, _, r23 = calculate_distances(spos, dihed_idx[:, :, [1, 2]], sbox)
            _, _, r34 = calculate_distances(spos, dihed_idx[:, :, [2, 3]], sbox)
        
            # Evaluate
            E, dihedral_forces = evaluate_torsion(
                r12,
                r23,
                r34,
                self.par.dihedral_params["map"][:, 0].unsqueeze(0).expand(nsystems,-1),
                self.par.dihedral_params["params"][param_idx],
                explicit_forces,
            )
            
            
            # for i in range(nsystems):
            #     pot[i]["dihedrals"] = pot[i]["dihedrals"] + E[i].sum()
                
            # Accumulate energies
            # Forces still need to be added per-system
    
            pot["dihedrals"] += E.sum(dim=1)  # E shape: (nsystems, ndihedrals)

            forces_flat = forces.view(-1, 3)
            offset = torch.arange(nsystems, device=pos.device) * pos.shape[1]  # (nsystems,)
        
            # Loop over each of the 4 atoms in the torsion
            for j in range(4):
                idx = dihed_idx[:, :, j] + offset[:, None]  # (nsystems, ndihedrals)
                idx = idx.reshape(-1)  # flatten to (nsystems * ndihedrals,)
                fvec = dihedral_forces[j].reshape(-1, 3)  # (nsystems * ndihedrals, 3)
                forces_flat.index_add_(0, idx, fvec)
    
    
            # for i in range(nsystems):
            #     # pot[i]["dihedrals"] = pot[i]["dihedrals"] + E[i].sum()
            #     forces[i].index_add_(0, dihed_idx[i, :, 0], dihedral_forces[0][i])
            #     forces[i].index_add_(0, dihed_idx[i, :, 1], dihedral_forces[1][i])
            #     forces[i].index_add_(0, dihed_idx[i, :, 2], dihedral_forces[2][i])
            #     forces[i].index_add_(0, dihed_idx[i, :, 3], dihedral_forces[3][i])
            

        if self.require_distances and len(self.ava_idx):
            if "repulsioncg" in self.energies:
            
                ava_idx = self.ava_idx
                ava_idx = ava_idx.unsqueeze(0).expand(nsystems,-1,-1)
                # # print('ava_idx shape', ava_idx.shape)
                nb_dist, nb_unitvec, _ = calculate_distances(spos, ava_idx, sbox)
                mapped_atom_types = self.par.mapped_atom_types.unsqueeze(0).expand(nsystems, -1)
        
                E, force_coeff = evaluate_repulsion_CG(
                    nb_dist,
                    ava_idx,
                    mapped_atom_types,
                    self.par.B,
                    explicit_forces,
                )
    
                # pot["repulsioncg"] += E.sum(dim=1)  # shape: (nsystems,)
                pot["repulsioncg"] = pot["repulsioncg"] + E.sum(dim=1)

        
                # forcevec: (nsystems, npairs, 3)
                forcevec = nb_unitvec * force_coeff.unsqueeze(-1)
                
                # Flatten system-wise forces
                forces_flat = forces.view(-1, 3)  # (nsystems * natoms, 3)
                
                # Offset atom indices to map each system into flat index space
                offset = torch.arange(nsystems, device=pos.device) * pos.shape[1]  # (nsystems,)
                idx0 = ava_idx[:, :, 0] + offset[:, None]  # (nsystems, npairs)
                idx1 = ava_idx[:, :, 1] + offset[:, None]
                
                # Flatten for index_add_
                idx0 = idx0.reshape(-1)
                idx1 = idx1.reshape(-1)
                fvec = forcevec.reshape(-1, 3)
                
                # Add to forces (minus and plus directions)
                forces_flat.index_add_(0, idx0, -fvec)
                forces_flat.index_add_(0, idx1, fvec)

        # for i in range(nsystems):             
        #     # pot[i]["repulsioncg"] = pot[i]["repulsioncg"] + E[i].sum()
        #     # if explicit_forces:

        #     forcevec = nb_unitvec[i] * force_coeff[i].unsqueeze(-1)  # (npairs, 3)
        #     forces[i].index_add_(0, ava_idx[i, :, 0], -forcevec)
        #     forces[i].index_add_(0, ava_idx[i, :, 1], forcevec)

        # for i in range(nsystems):
        #     spos = pos[i]
        #     sbox = box[i][torch.eye(3).bool()]  # Use only the diagonal

        #     if "angles" in self.energies and self.par.angle_params is not None:
        #         angle_idx = self.par.angle_params["idx"]
        #         param_idx = self.par.angle_params["map"][:, 1]
        #         _, _, r21 = calculate_distances(spos, angle_idx[:, [0, 1]], sbox)
        #         _, _, r23 = calculate_distances(spos, angle_idx[:, [2, 1]], sbox)
        #         E, angle_forces = evaluate_angles(
        #             r21,
        #             r23,
        #             self.par.angle_params["params"][param_idx],
        #             explicit_forces,
        #         )

        #         pot[i]["angles"] = pot[i]["angles"] + E.sum()
        #         if explicit_forces:
        #             forces[i].index_add_(0, angle_idx[:, 0], angle_forces[0])
        #             forces[i].index_add_(0, angle_idx[:, 1], angle_forces[1])
        #             forces[i].index_add_(0, angle_idx[:, 2], angle_forces[2])

        #     # if "dihedrals" in self.energies and self.par.dihedral_params is not None:
        #     #     dihed_idx = self.par.dihedral_params["idx"]
        #     #     dihed_idx = dihed_idx.unsqueeze(0).expand(nsystems, -1, -1)
        #     #     param_idx = self.par.dihedral_params["map"][:, 1]
        #     #     param_idx = param_idx.unsqueeze(0).expand(nsystems, -1)
                
        #     #     _, _, r12 = calculate_distances(spos, dihed_idx[:, [0, 1]], sbox)
        #     #     _, _, r23 = calculate_distances(spos, dihed_idx[:, [1, 2]], sbox)
        #     #     _, _, r34 = calculate_distances(spos, dihed_idx[:, [2, 3]], sbox)
        #     #     E, dihedral_forces = evaluate_torsion(
        #     #         r12,
        #     #         r23,
        #     #         r34,
        #     #         self.par.dihedral_params["map"][:, 0].unsqueeze(0).expand(nsystems,-1),
        #     #         self.par.dihedral_params["params"][param_idx],
        #     #         explicit_forces,
        #     #     )

        #     #     pot[i]["dihedrals"] = pot[i]["dihedrals"] + E.sum()
        #     #     if explicit_forces:
        #     #         forces[i].index_add_(0, dihed_idx[:, 0], dihedral_forces[0])
        #     #         forces[i].index_add_(0, dihed_idx[:, 1], dihedral_forces[1])
        #     #         forces[i].index_add_(0, dihed_idx[:, 2], dihedral_forces[2])
        #     #         forces[i].index_add_(0, dihed_idx[:, 3], dihedral_forces[3])

        #     if "1-4" in self.energies and self.par.nonbonded_14_params is not None:
        #         idx14 = self.par.nonbonded_14_params["idx"]
        #         nb_dist, nb_unitvec, _ = calculate_distances(spos, idx14, sbox)

        #         nonbonded_14_params = self.par.nonbonded_14_params["params"]

        #         if self.cutoff is not None:
        #             (
        #                 nb_dist,
        #                 nb_unitvec,
        #                 nonbonded_14_params,
        #                 idx14,
        #             ) = self._filter_by_cutoff(
        #                 nb_dist,
        #                 (
        #                     nb_dist,
        #                     nb_unitvec,
        #                     self.par.nonbonded_14_params,
        #                     self.par.idx14,
        #                 ),
        #             )
        #         prm_idx = self.par.nonbonded_14_params["map"][:, 1]
        #         aa = nonbonded_14_params[prm_idx, 0]
        #         bb = nonbonded_14_params[prm_idx, 1]
        #         scnb = nonbonded_14_params[prm_idx, 2]
        #         scee = nonbonded_14_params[prm_idx, 3]

        #         if "lj" in self.energies:
        #             E, force_coeff = evaluate_LJ_internal(
        #                 nb_dist, aa, bb, scnb, None, None, explicit_forces
        #             )
        #             pot[i]["lj"] = pot[i]["lj"] + E.sum()
        #             if explicit_forces:
        #                 forcevec = nb_unitvec * force_coeff[:, None]
        #                 forces[i].index_add_(0, idx14[:, 0], -forcevec)
        #                 forces[i].index_add_(0, idx14[:, 1], forcevec)
        #         if "electrostatics" in self.energies:
        #             E, force_coeff = evaluate_electrostatics(
        #                 nb_dist,
        #                 idx14,
        #                 self.par.charges,
        #                 scee,
        #                 cutoff=None,
        #                 rfa=False,
        #                 solventDielectric=self.solventDielectric,
        #                 explicit_forces=explicit_forces,
        #             )
        #             pot[i]["electrostatics"] = pot[i]["electrostatics"] + E.sum()
        #             if explicit_forces:
        #                 forcevec = nb_unitvec * force_coeff[:, None]
        #                 forces[i].index_add_(0, idx14[:, 0], -forcevec)
        #                 forces[i].index_add_(0, idx14[:, 1], forcevec)

        #     if "impropers" in self.energies and self.par.improper_params is not None:
        #         impr_idx = self.par.improper_params["idx"]
        #         param_idx = self.par.improper_params["map"][:, 1]
        #         _, _, r12 = calculate_distances(spos, impr_idx[:, [0, 1]], sbox)
        #         _, _, r23 = calculate_distances(spos, impr_idx[:, [1, 2]], sbox)
        #         _, _, r34 = calculate_distances(spos, impr_idx[:, [2, 3]], sbox)
        #         E, improper_forces = evaluate_torsion(
        #             r12,
        #             r23,
        #             r34,
        #             self.par.improper_params["map"][:, 0],
        #             self.par.improper_params["params"][param_idx],
        #             explicit_forces,
        #         )

        #         pot[i]["impropers"] = pot[i]["impropers"] + E.sum()
        #         if explicit_forces:
        #             forces[i].index_add_(0, impr_idx[:, 0], improper_forces[0])
        #             forces[i].index_add_(0, impr_idx[:, 1], improper_forces[1])
        #             forces[i].index_add_(0, impr_idx[:, 2], improper_forces[2])
        #             forces[i].index_add_(0, impr_idx[:, 3], improper_forces[3])

        #     # Non-bonded terms
        #     if self.require_distances and len(self.ava_idx):
        #         # Lazy mode: Do all vs all distances
        #         # TODO: These distance calculations are fucked once we do neighbourlists since they will vary per system!!!!
        #         nb_dist, nb_unitvec, _ = calculate_distances(spos.unsqueeze(0), self.ava_idx.unsqueeze(0), sbox.unsqueeze(0))
        #         ava_idx = self.ava_idx
        #         if self.cutoff is not None:
        #             nb_dist, nb_unitvec, ava_idx = self._filter_by_cutoff(
        #                 nb_dist, (nb_dist, nb_unitvec, ava_idx)
        #             )

        #         for v in self.energies:
        #             if v == "electrostatics":
        #                 E, force_coeff = evaluate_electrostatics(
        #                     nb_dist,
        #                     ava_idx,
        #                     self.par.charges,
        #                     cutoff=self.cutoff,
        #                     rfa=self.rfa,
        #                     solventDielectric=self.solventDielectric,
        #                     explicit_forces=explicit_forces,
        #                 )
        #                 pot[i][v] = pot[i][v] + E.sum()
        #             elif v == "lj":
        #                 E, force_coeff = evaluate_LJ(
        #                     nb_dist,
        #                     ava_idx,
        #                     self.par.mapped_atom_types,
        #                     self.par.A,
        #                     self.par.B,
        #                     self.switch_dist,
        #                     self.cutoff,
        #                     explicit_forces,
        #                 )
        #                 pot[i][v] = pot[i][v] + E.sum()
        #             elif v == "repulsion":
        #                 E, force_coeff = evaluate_repulsion(
        #                     nb_dist,
        #                     ava_idx,
        #                     self.par.mapped_atom_types,
        #                     self.par.A,
        #                     explicit_forces,
        #                 )
        #                 pot[i][v] = pot[i][v] + E.sum()
        #             # elif v == "repulsioncg":
        #             #     E, force_coeff = evaluate_repulsion_CG(
        #             #         nb_dist,
        #             #         ava_idx,
        #             #         self.par.mapped_atom_types,
        #             #         self.par.B,
        #             #         explicit_forces,
        #             #     )
        #             #     pot[i][v] = pot[i][v] + E.sum()
        #             else:
        #                 continue

        #             if explicit_forces:
        #                 forcevec = nb_unitvec * force_coeff[:, None]
        #                 forces[i].index_add_(0, ava_idx[:, 0], -forcevec)
        #                 forces[i].index_add_(0, ava_idx[:, 1], forcevec)

        if self.external:
        # ext_ene, ext_force = self.external.calculate(pos, box)
        # for s in range(nsystems):
        #     pot[s]["external"] = pot[s]["external"] + ext_ene[s]
        # # if explicit_forces:
        # forces += ext_force

            ext_ene, ext_force = self.external.calculate(pos, box)  # ext_ene: (nsystems,), ext_force: (nsystems, natoms, 3)
    
            ext_ene = ext_ene.view(-1)  # Ensures shape is (nsystems,)
            
            pot["external"] += ext_ene  # batched addition
            forces += ext_force         # batched addition (forces must be tensor, not list)

        if not explicit_forces and calculateForces:
            enesum = torch.zeros(1, device=pos.device, dtype=pos.dtype)
            for i in range(nsystems):
                for ene in pot[i]:
                    if pot[i][ene].requires_grad:
                        enesum = enesum + pot[i][ene]
            forces[:] = -torch.autograd.grad(
                enesum, pos, only_inputs=True, retain_graph=True
            )[0]

        if not returnDetails:
            # pot = torch.stack([torch.sum(torch.cat(list(pp.values()))) for pp in pot])
            # total energy per system
            pot = torch.stack([v for v in pot.values()], dim=1).sum(dim=1)  # shape: (nsystems,)

        # if toNumpy:
        #     if returnDetails:
        #         return [{k: v.cpu().item() for k, v in pp.items()} for pp in pot]
        #     else:
        #         return [pp.cpu().item() for pp in pot]

        if toNumpy:
            if returnDetails:
                if return_forces:
                    return [{k: pot[k][i].cpu().item() for k in pot} for i in range(nsystems)], forces
                else:
                    return [{k: pot[k][i].cpu().item() for k in pot} for i in range(nsystems)]
            else:
                if return_forces:
                    return pot.cpu().numpy().tolist(), forces
                else:
                    return pot.cpu().numpy().tolist()

        if return_forces:
            return pot, forces
        else:      
            return pot

    def _make_indeces(self, natoms, excludepairs, device):
        fullmat = np.full((natoms, natoms), True, dtype=bool)
        if len(excludepairs):
            excludepairs = np.array(excludepairs)
            fullmat[excludepairs[:, 0], excludepairs[:, 1]] = False
            fullmat[excludepairs[:, 1], excludepairs[:, 0]] = False
        fullmat = np.triu(fullmat, +1)
        allvsall_indeces = np.vstack(np.where(fullmat)).T
        ava_idx = torch.tensor(allvsall_indeces).to(device)
        return ava_idx


def wrap_dist(dist, box):  # dist: (nsystems, nbonds, 3), box: (nsystems, 3)
    if box is None or torch.all(box == 0):
        return dist
    # Ensure box is broadcastable to (nsystems, nbonds, 3)
    box = box[:, None, :]  # (nsystems, 1, 3)
    wdist = dist - box * torch.round(dist / box)
    return wdist

def calculate_distances(atom_pos, atom_idx, box):
    """
    atom_pos: (nsystems, natoms, 3)
    atom_idx: (nsystems, nbonds, 2)
    box: (nsystems, 3)
    """
    # Gather atom positions for each system and bond
    src = torch.gather(atom_pos, 1, atom_idx[..., 0].unsqueeze(-1).expand(-1, -1, 3))  # (nsystems, nbonds, 3)
    dst = torch.gather(atom_pos, 1, atom_idx[..., 1].unsqueeze(-1).expand(-1, -1, 3))

    direction_vec = wrap_dist(src - dst, box)  # (nsystems, nbonds, 3)
    dist = torch.norm(direction_vec, dim=-1, keepdim=True)            # (nsystems, nbonds)
    direction_unitvec = direction_vec / dist  # (nsystems, nbonds, 3)

    return dist.squeeze(), direction_unitvec, direction_vec


ELEC_FACTOR = 1 / (4 * const.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge**2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


def evaluate_LJ(
    dist, pair_indeces, atom_types, A, B, switch_dist, cutoff, explicit_forces=True
):
    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    bb = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    return evaluate_LJ_internal(dist, aa, bb, 1, switch_dist, cutoff, explicit_forces)


def evaluate_LJ_internal(
    dist, aa, bb, scale, switch_dist, cutoff, explicit_forces=True
):
    force = None

    rinv1 = 1 / dist
    rinv6 = rinv1**6
    rinv12 = rinv6 * rinv6

    pot = ((aa * rinv12) - (bb * rinv6)) / scale
    if explicit_forces:
        force = (-12 * aa * rinv12 + 6 * bb * rinv6) * rinv1 / scale

    # Switching function
    if switch_dist is not None and cutoff is not None:
        mask = dist > switch_dist
        t = (dist[mask] - switch_dist) / (cutoff - switch_dist)
        switch_val = 1 + t * t * t * (-10 + t * (15 - t * 6))
        if explicit_forces:
            switch_deriv = t * t * (-30 + t * (60 - t * 30)) / (cutoff - switch_dist)
            force[mask] = (
                switch_val * force[mask] + pot[mask] * switch_deriv / dist[mask]
            )
        pot[mask] = pot[mask] * switch_val

    return pot, force


def evaluate_repulsion(
    dist, pair_indeces, atom_types, A, scale=1, explicit_forces=True
):  # LJ without B
    force = None

    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1**6
    rinv12 = rinv6 * rinv6

    pot = (aa * rinv12) / scale
    if explicit_forces:
        force = (-12 * aa * rinv12) * rinv1 / scale
    return pot, force


# def evaluate_repulsion_CG(
#     dist, pair_indeces, atom_types, B, scale=1, explicit_forces=True
# ):  # Repulsion like from CGNet
#     force = None

#     atomtype_indices = atom_types[pair_indeces]
#     coef = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

#     rinv1 = 1 / dist
#     rinv6 = rinv1**6

#     pot = (coef * rinv6) / scale
#     if explicit_forces:
#         force = (-6 * coef * rinv6) * rinv1 / scale
#     return pot, force

def evaluate_repulsion_CG(
    dist, pair_indices, atom_types, B, scale=1.0, explicit_forces=True
):
    """
    Repulsion like from CGNet
    
    dist: (nsystems, npairs)
    pair_indices: (nsystems, npairs, 2)
    atom_types: (nsystems, natoms)
    B: (ntypes, ntypes)
    Returns:
        pot: (nsystems, npairs)
        force: (nsystems, npairs) or None
    """
    nsystems, npairs, _ = pair_indices.shape

    # Get atom type indices for each pair
    idx0 = pair_indices[..., 0]  # (nsystems, npairs)
    idx1 = pair_indices[..., 1]  # (nsystems, npairs)

    # print(atom_types.shape)

    type0 = torch.gather(atom_types, 1, idx0)  # (nsystems, npairs)
    type1 = torch.gather(atom_types, 1, idx1)  # (nsystems, npairs)

    # Look up B coefficients
    coef = B[type0, type1]  # uses advanced indexing (nsystems, npairs)

    rinv1 = 1.0 / dist
    rinv6 = rinv1 ** 6

    pot = (coef * rinv6) / scale

    # if explicit_forces:
    force = (-6 * coef * rinv6 * rinv1) / scale
    # else:
        # force = None

    return pot, force
    
def evaluate_electrostatics(
    dist,
    pair_indeces,
    atom_charges,
    scale=1,
    cutoff=None,
    rfa=False,
    solventDielectric=78.5,
    explicit_forces=True,
):
    force = None
    if rfa:  # Reaction field approximation for electrostatics with cutoff
        # http://docs.openmm.org/latest/userguide/theory.html#coulomb-interaction-with-cutoff
        # Ilario G. Tironi, René Sperb, Paul E. Smith, and Wilfred F. van Gunsteren. A generalized reaction field method
        # for molecular dynamics simulations. Journal of Chemical Physics, 102(13):5451–5459, 1995.
        denom = (2 * solventDielectric) + 1
        krf = (1 / cutoff**3) * (solventDielectric - 1) / denom
        crf = (1 / cutoff) * (3 * solventDielectric) / denom
        common = (
            ELEC_FACTOR
            * atom_charges[pair_indeces[:, 0]]
            * atom_charges[pair_indeces[:, 1]]
            / scale
        )
        dist2 = dist**2
        pot = common * ((1 / dist) + krf * dist2 - crf)
        if explicit_forces:
            force = common * (2 * krf * dist - 1 / dist2)
    else:
        pot = (
            ELEC_FACTOR
            * atom_charges[pair_indeces[:, 0]]
            * atom_charges[pair_indeces[:, 1]]
            / dist
            / scale
        )
        if explicit_forces:
            force = -pot / dist
    return pot, force


def evaluate_bonds(dist, bond_params, explicit_forces=True):
    """
    dist: (nsystems, nbonds)
    bond_params: (nsystems, nbonds, 2)
    """
    k0 = bond_params[..., 0]  # (nsystems, nbonds)
    d0 = bond_params[..., 1]
    x = dist - d0
    pot = k0 * (x ** 2)
    force = 2 * k0 * x #if explicit_forces else None
    return pot, force


def evaluate_angles(r21, r23, angle_params, explicit_forces=True):
    k0 = angle_params[:, 0]
    theta0 = angle_params[:, 1]

    dotprod = torch.sum(r23 * r21, dim=1)
    norm23inv = 1 / torch.norm(r23, dim=1)
    norm21inv = 1 / torch.norm(r21, dim=1)

    cos_theta = dotprod * norm21inv * norm23inv
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)

    delta_theta = theta - theta0
    pot = k0 * delta_theta * delta_theta

    force0, force1, force2 = None, None, None
    if explicit_forces:
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        coef = torch.zeros_like(sin_theta)
        nonzero = sin_theta != 0
        coef[nonzero] = -2.0 * k0[nonzero] * delta_theta[nonzero] / sin_theta[nonzero]
        force0 = (
            coef[:, None]
            * (cos_theta[:, None] * r21 * norm21inv[:, None] - r23 * norm23inv[:, None])
            * norm21inv[:, None]
        )
        force2 = (
            coef[:, None]
            * (cos_theta[:, None] * r23 * norm23inv[:, None] - r21 * norm21inv[:, None])
            * norm23inv[:, None]
        )
        force1 = -(force0 + force2)

    return pot, (force0, force1, force2)


def batched_scatter_add(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Performs batched scatter_add on the last dimension, assuming:
    - values: (B, N)
    - indices: (B, N), with indices per batch in [0, M)
    
    Returns:
    - result: (B, M), where M = max index + 1
    """
    B, N = values.shape
    M = indices.max().item() + 1

    # Flatten and offset batch indices
    flat_values = values.flatten()  # (B * N)
    flat_indices = indices + torch.arange(B, device=values.device).unsqueeze(1) * M
    flat_indices = flat_indices.flatten()

    # Flat output tensor
    result_flat = torch.zeros(B * M, dtype=values.dtype, device=values.device)
    result_flat.scatter_add_(0, flat_indices, flat_values)

    return result_flat.view(B, M)

def evaluate_torsion(r12, r23, r34, dih_idx, torsion_params, explicit_forces=True):
    # Cross products
    
    crossA = torch.cross(r12, r23, dim=-1)
    crossB = torch.cross(r23, r34, dim=-1)
    crossC = torch.cross(r23, crossA, dim=-1)

    # Norms
    normA = torch.norm(crossA, dim=-1)
    normB = torch.norm(crossB, dim=-1)
    normC = torch.norm(crossC, dim=-1)

    # Angle phi
    normcrossB = crossB / normB.unsqueeze(-1)
    cosPhi = torch.sum(crossA * normcrossB, dim=-1) / normA
    sinPhi = torch.sum(crossC * normcrossB, dim=-1) / normC
    phi = -torch.atan2(sinPhi, cosPhi)

    k0 = torsion_params[..., 0]
    phi0 = torsion_params[..., 1]
    per = torsion_params[..., 2]

    pot = torch.zeros(k0.shape, dtype=r12.dtype, layout=r12.layout, device=r12.device)

    if explicit_forces:
        coeff = pot = torch.zeros(k0.shape, 
                                  dtype=r12.dtype,
                                  layout=r12.layout,
                                  device=r12.device
                                 )

    # Assume all systems share the same AMBER/CHARMM mask
    amber_mask = per[0] > 0  # shape: (ndihedrals,)
    charmm_mask = ~amber_mask

    # Index phi using dih_idx: shape (nsystems, ndihedrals)
    phi_dih = torch.gather(phi, dim=1, index=dih_idx)
    # print(phi_dih.shape)

    ### AMBER torsions
    if amber_mask.all():

        k0_amber     = k0#[:, amber_mask]
        phi0_amber   = phi0#[:, amber_mask]
        per_amber    = per#[:, amber_mask]
        phi_amber    = phi_dih#[:, amber_mask]
        angleDiff   = per_amber * phi_amber - phi0_amber
        # pot_amber    = k0_amber * (1 + torch.cos(angleDiff))

        # values: (nsystems, ntorsions)
        # dih_idx: (nsystems, ntorsions) with target indices in [0, ntargets)
        
        pot_amber = k0_amber * (1 + torch.cos(angleDiff))   # (B, N)
        pot = batched_scatter_add(pot_amber, dih_idx)  # (B, M)
        # print(pot.sum())
        
        if explicit_forces:
            coeff_amber = -per_amber * k0_amber * torch.sin(angleDiff)
            coeff = batched_scatter_add(coeff_amber, dih_idx)


    ## CHARMM torsions
    if charmm_mask.any():
        k0_charmm = k0[:, charmm_mask]
        phi0_charmm = phi0[:, charmm_mask]
        dih_idx_charmm = dih_idx[:, charmm_mask]
        phi_charmm = phi_dih[:, charmm_mask]

        angleDiff = phi_charmm - phi0_charmm
        angleDiff = torch.remainder(angleDiff + torch.pi, 2 * torch.pi) - torch.pi
        # pot.scatter_add_(1, dih_idx_charmm, k0_charmm * angleDiff**2)

        pot_charmm = k0_charmm * angleDiff**2
        pot[:, charmm_mask] = pot_charmm  # assign directly instead of scatter

        if explicit_forces:
            coeff.scatter_add_(1, dih_idx_charmm, 2 * k0_charmm * angleDiff)
    
    ### Forces
    force0_red, force1_red, force2_red, force3_red = None, None, None, None
    if explicit_forces:
        # Compute norms per bond
        normDelta2 = torch.norm(r23, dim=-1)                     # [nsystems, nbonds]
        # norm2Delta2 = normDelta2 ** 2                            # [nsystems, nbonds]
    
        # Gather values per torsion
        normDelta2_dih = normDelta2#.gather(1, dih_idx)           # [nsystems, ntorsions]
        # norm2Delta2_dih = norm2Delta2.gather(1, dih_idx)         # [nsystems, ntorsions]
        norm2Delta2_dih = normDelta2_dih ** 2 
        normA_dih = normA#.gather(1, dih_idx)                     # [nsystems, ntorsions]
        normB_dih = normB#.gather(1, dih_idx)                     # [nsystems, ntorsions]
        
        # Dot products gathered per torsion
        dot12 = torch.sum(r12 * r23, dim=-1)#.gather(1, dih_idx)
        dot34 = torch.sum(r34 * r23, dim=-1)#.gather(1, dih_idx)
    
        # Force factors
        forceFactor0 = (-coeff * normDelta2_dih) / (normA_dih ** 2)
        forceFactor1 = dot12 / norm2Delta2_dih
        forceFactor2 = dot34 / norm2Delta2_dih
        forceFactor3 = (coeff * normDelta2_dih) / (normB_dih ** 2)
        
        # Gather cross products per torsion
        crossA_dih = crossA#.gather(1, dih_idx.unsqueeze(-1).expand(-1, -1, 3))
        crossB_dih = crossB#.gather(1, dih_idx.unsqueeze(-1).expand(-1, -1, 3))
    
        # Force vectors
        force0vec = forceFactor0.unsqueeze(-1) * crossA_dih
        force3vec = forceFactor3.unsqueeze(-1) * crossB_dih
        s = forceFactor1.unsqueeze(-1) * force0vec - forceFactor2.unsqueeze(-1) * force3vec
        
        # Final forces
        force0 = -force0vec
        force1 = force0vec + s
        force2 = force3vec - s
        force3 = -force3vec

    return pot, (force0, force1, force2, force3)


# def evaluate_torsion(r12, r23, r34, dih_idx, torsion_params, explicit_forces=True):
#     # Calculate dihedral angles from vectors
#     crossA = torch.cross(r12, r23, dim=1)
#     crossB = torch.cross(r23, r34, dim=1)
#     crossC = torch.cross(r23, crossA, dim=1)
#     normA = torch.norm(crossA, dim=1)
#     normB = torch.norm(crossB, dim=1)
#     normC = torch.norm(crossC, dim=1)
#     normcrossB = crossB / normB.unsqueeze(1)
#     cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
#     sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
#     phi = -torch.atan2(sinPhi, cosPhi)

#     ntorsions = r12.shape[0]
#     pot = torch.zeros(ntorsions, dtype=r12.dtype, layout=r12.layout, device=r12.device)
#     if explicit_forces:
#         coeff = torch.zeros(
#             ntorsions, dtype=r12.dtype, layout=r12.layout, device=r12.device
#         )

#     k0 = torsion_params[:, 0]
#     phi0 = torsion_params[:, 1]
#     per = torsion_params[:, 2]

#     if torch.all(per > 0):  # AMBER torsions
#         angleDiff = per * phi[dih_idx] - phi0
#         pot = torch.scatter_add(pot, 0, dih_idx, k0 * (1 + torch.cos(angleDiff)))
#         if explicit_forces:
#             coeff = torch.scatter_add(coeff, 0, dih_idx, -per * k0 * torch.sin(angleDiff))
#     else:  # CHARMM torsions
#         angleDiff = phi[dih_idx] - phi0
#         angleDiff[angleDiff < -pi] = angleDiff[angleDiff < -pi] + 2 * pi
#         angleDiff[angleDiff > pi] = angleDiff[angleDiff > pi] - 2 * pi
#         pot = torch.scatter_add(pot, 0, dih_idx, k0 * angleDiff**2)
#         if explicit_forces:
#             coeff = torch.scatter_add(coeff, 0, dih_idx, 2 * k0 * angleDiff)

#     # coeff.unsqueeze_(1)

#     force0, force1, force2, force3 = None, None, None, None
#     if explicit_forces:
#         # Taken from OpenMM
#         normDelta2 = torch.norm(r23, dim=1)
#         norm2Delta2 = normDelta2**2
#         forceFactor0 = (-coeff * normDelta2) / (normA**2)
#         forceFactor1 = torch.sum(r12 * r23, dim=1) / norm2Delta2
#         forceFactor2 = torch.sum(r34 * r23, dim=1) / norm2Delta2
#         forceFactor3 = (coeff * normDelta2) / (normB**2)

#         force0vec = forceFactor0.unsqueeze(1) * crossA
#         force3vec = forceFactor3.unsqueeze(1) * crossB
#         s = (
#             forceFactor1.unsqueeze(1) * force0vec
#             - forceFactor2.unsqueeze(1) * force3vec
#         )

#         force0 = -force0vec
#         force1 = force0vec + s
#         force2 = force3vec - s
#         force3 = -force3vec

#     return pot, (force0, force1, force2, force3)