import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import fiumd

class WrapperEnsemble(nn.Module):
    def __init__(self, ensemble: fiumd.ensembles.Ensemble):
        super().__init__()
        self.ensemble = ensemble
        self.groups_per_system = []
        self.nongrouped_per_system = []

        for system in ensemble.systems:
            groups, nongrouped = self._calculate_molecule_groups(
                natoms=system.natoms,
                bonds=system.molecule.bonds,
                device=ensemble.device
            )
            self.groups_per_system.append(groups)
            self.nongrouped_per_system.append(nongrouped)

    def forward(self, wrapidx_per_system=None):
        """
        Applies center-of-geometry wrapping per system.
        Optionally recenters using a group (e.g. solute) per system.

        Parameters
        ----------
        wrapidx_per_system : list of list[int] or None
            Local atom indices (per system) of centering group (e.g. solute).
        """
        pos = self.ensemble.positions
        box = self.ensemble.box

        for i in range(self.ensemble.nsystems):
            offset = self.ensemble.atom_offsets[i]
            natoms = self.ensemble.natoms_per_system[i]
            box_i = box[i]
            pos_i = pos[offset:offset + natoms]

            # Optional centering by COM
            if wrapidx_per_system is not None and wrapidx_per_system[i] is not None:
                com = pos_i[wrapidx_per_system[i]].mean(dim=0)
                pos_i.sub_(com).add_(box_i / 2)

            for group in self.groups_per_system[i]:
                tmp_com = pos_i[group].mean(dim=0)
                offset_vec = torch.floor(tmp_com / box_i) * box_i
                pos_i[group] -= offset_vec

            if len(self.nongrouped_per_system[i]):
                ng = self.nongrouped_per_system[i]
                offset_vec = torch.floor(pos_i[ng] / box_i) * box_i
                pos_i[ng] -= offset_vec

    def _calculate_molecule_groups(self, natoms, bonds, device):
        if bonds is not None and len(bonds):
            bondGraph = nx.Graph()
            bondGraph.add_nodes_from(range(natoms))
            bondGraph.add_edges_from(bonds.astype(np.int64))

            molgroups = list(nx.connected_components(bondGraph))

            nongrouped = torch.tensor(
                [list(g)[0] for g in molgroups if len(g) == 1],
                device=device
            )

            molgroups = [
                torch.tensor(list(g), device=device)
                for g in molgroups if len(g) > 1
            ]
        else:
            molgroups = []
            nongrouped = torch.arange(natoms, device=device)

        return molgroups, nongrouped