# FiuMD

FiuMD runs many coarse-grained protein simulations of different size and
topology as one GPU batch. It is a fork of
[TorchMD](https://github.com/torchmd/torchmd). Each classical term (bonds,
repulsion, dihedrals) acts on the whole batch with per-system parameters. One
shared [TorchMD-Net](https://github.com/torchmd/torchmd-net) call returns a
neural-network energy correction for every system, and the integrators advance
each trajectory with its own timestep, box, cutoff and thermostat parameters.

Units follow TorchMD: kcal/mol, K, g/mol and Å.

## Installation

Tested with Python 3.10.18, PyTorch 2.13.0+cu130 (CUDA 13.0), TorchMD-Net 3.0.3
and TorchMD 1.1.2 on Linux.

```bash
mamba create -n fiumd -c conda-forge -c acellera python=3.10 moleculekit parmed
mamba activate fiumd
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install torchmd-net==3.0.3
git clone https://github.com/hernandezhdd/fiumd.git
pip install -e fiumd
```

TorchMD itself (`pip install torchmd==1.1.2`) is needed only to run the
reference comparisons in `paper/`.

## Usage

To advance two proteins in one batch, prepare one `(molecule, parameters, box)`
triple per system: a MoleculeKit `Molecule`, its FiuMD `Parameters` and a tensor
of three box lengths in Å. Then concatenate positions and velocities in the
same system and bead order and build an `Ensemble`. The paper gives a complete
example. `paper/benchmark/heterogeneous_ensemble.py` builds mixed batches of
four proteins from the command line.

`examples/two_proteins.py` runs chignolin and Trp-cage in one batch at two
temperatures (inputs from `paper/datos`, see `paper/datos/README.md`).

## Reproducing the paper

`paper/` has the drivers, figure scripts and inputs for the arXiv v1 paper. Start
with [`paper/README.md`](paper/README.md).

## Citation

If you use FiuMD, please cite the FiuMD paper (see `CITATION.cff`; the arXiv
identifier will be added on posting) and TorchMD:

```bibtex
@article{doerr2021torchmd,
  title   = {TorchMD: A Deep Learning Framework for Molecular Simulations},
  author  = {Doerr, Stefan and Majewski, Maciej and P{\'e}rez, Adri{\`a} and Kr{\"a}mer, Andreas
             and Clementi, Cecilia and No{\'e}, Frank and Giorgino, Toni and De Fabritiis, Gianni},
  journal = {Journal of Chemical Theory and Computation},
  volume  = {17},
  pages   = {2355--2363},
  year    = {2021},
  doi     = {10.1021/acs.jctc.0c01343}
}
```

## License

MIT (see `LICENSE.md`), as in upstream TorchMD. Some file-format readers come
from MoleculeKit, which has its own license; see
<https://github.com/Acellera/moleculekit>.

## Acknowledgements

FiuMD builds on TorchMD by Acellera and the TorchMD authors.