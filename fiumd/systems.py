import torch
import torch.nn as nn
# from fiumd.forces import Force
import fiumd.forces
import fiumd.utils
import fiumd.parameters
import numpy as np
from typing import Type
import moleculekit.molecule
from typing import Optional

class System(nn.Module):
    def __init__(self,
                 molecule: Optional[moleculekit.molecule.Molecule],
                 force_terms: list[str],
                 # params: dict[str, dict[str, torch.Tensor]],
                 params: fiumd.parameters.Parameters,
                 box,
                 cutoff: Optional[float] = None,
                ):
        """
        Parameters
        ----------

        box : torch.Tensor, shape (3,)
        molecule: molecule in system (moleculekit.molecule.Molecule)
        force_terms : force term present. list of str with the names of fiumd.forces.Force subclasses
            For example: {'BondForce', RepulsionCGForce}
        params: fiumd.parameters.Parameters
        cutoff : float or None, optional
            Cutoff distance (in Å) for nonbonded interactions in this system (default: None).
        """
        super().__init__()

        if cutoff is not None:
            if isinstance(cutoff, torch.Tensor):
                if cutoff.ndim != 0:
                    raise ValueError(f"System cutoff must be a scalar, got tensor of shape {tuple(cutoff.shape)}")
                cutoff = float(cutoff.item())
            elif isinstance(cutoff, (float, int)):
                cutoff = float(cutoff)
            else:
                raise TypeError(f"cutoff must be a float, int, scalar Tensor, or None, got {type(cutoff)}")
            if cutoff <= 0:
                raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.molecule = molecule
        self.box = nn.parameter.Buffer(box)               # (3,)
        self.force_terms = force_terms  # per-interaction type
        self.params = params
        self.cutoff = cutoff
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
    

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)
