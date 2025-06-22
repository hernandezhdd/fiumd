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
    

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)
        