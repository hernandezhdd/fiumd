# Reproducing the arXiv v1 paper

Scripts and inputs behind *FiuMD: heterogeneous batching for coarse-grained
molecular dynamics with neural network potentials* (arXiv v1).

All runs in the paper used the library code of this repository at revision
`433cedf` (tag `v0.1.0`). Each run's `config.json` records that commit, the GPU
and the torch build.

## Layout

```
paper/
  benchmark/   drivers (simulations, tests, timing) and figure scripts
  datos/       inputs, one folder per protein (see datos/README.md)
  results/     raw outputs; not in git, download from Zenodo (see below)
```

Run every command from `paper/`. The scripts resolve `datos/` and `results/`
relative to this folder.

## Environment

Tested stack: Python 3.10.18, PyTorch 2.13.0+cu130 (CUDA 13.0), TorchMD 1.1.2,
TorchMD-Net 3.0.3, MoleculeKit, on one NVIDIA RTX 5060 Ti. TorchMD is needed
only for the reference runs (`--engine torchmd`, the B=1 force checks and the
`torchmd_seq` timing mode).

```bash
pip install -e ..            # FiuMD, from the repository root
pip install torchmd==1.1.2 scipy matplotlib pandas
```

## Inputs

```bash
python datos/download_data.py                          # ~45 MB, a few minutes
python benchmark/check_benchmark_data.py --fix --deep  # writes model_modified.ckpt, checks every file
```

`datos/README.md` lists the source and MD5 checksum of every input file. Only
the lambda-repressor checkpoint is used: all runs pass `--nnp lambda`.

## Figures

| Fig. | Content | Driver(s) → raw output | Figure script → PNG |
|---|---|---|---|
| 3 | Energy and force agreement in a mixed batch (float64) | `test_independencie.py` → `results/a2_nve/run5/` | `fig_a2_paper.py` → `energy_all_nnp.png`, `forces_all_nnp.png` |
| 4 | Deterministic trajectories, image contacts | `test_multiple_systems.py` → `results/a2_nve/run3/nve/` | `fig_a2_paper.py` → `nve_images_fiumd.png`, `nve_final.png` |
| 5, 6 | Mean and width of the kinetic temperature | `run_a3_escalera_fiumd.sh` and the TorchMD loop below → `results/sistemas_multiples/a3_*` | `fig_a3_paper.py` → `a3_mean_T.png`, `a3_sigma.png` |
| 7 | Wall time and speedup of a mixed job | `bench_heterogeneous_ensemble.py` → `results/b1/` | `fig_b1_paper.py` → `b1_step_time.png`, `b1_speedup.png` |

Figs. 1 and 2 are schematics and use no data.

### Fig. 3: energies and forces

```bash
python benchmark/test_independencie.py --composition cln:32,trp:16,bbl:8,lambda:4 \
    --boxes data report tight --nnp lambda --precision float64 --seed 0 \
    --outdir results/a2_nve/run5
```

### Fig. 4: deterministic trajectories

```bash
python benchmark/test_multiple_systems.py --composition cln:32,trp:16,bbl:8,lambda:4 \
    --boxes report images --images_composition cln:3,trp:16,bbl:8,lambda:4 \
    --steps 1000 --temps cln:300,trp:350,bbl:370,lambda:400 \
    --nnp lambda --precision float64 --seed 0 --outdir results/a2_nve/run3/nve
```

Figures 3 and 4 together

```bash
python benchmark/fig_a2_paper.py --fuerzas results/a2_nve/run5 \
    --nve results/a2_nve/run3/nve --outdir results/a2_nve/figures
```

### Figs. 5 and 6: temperature ladder

FiuMD: four heterogeneous batches in a Latin square, each with the four
temperatures at once, one per protein (300 000 steps each):

```bash
bash benchmark/run_a3_escalera_fiumd.sh
```

TorchMD reference: one run per temperature, each started from a short FiuMD
run that writes the initial state:

```bash
for T in 300 340 380 420; do
  python -u benchmark/heterogeneous_ensemble.py --engine fiumd \
      --composition cln:8,trp:8,bbl:4,lambda:4 --nnp lambda --precision float32 \
      --temp $T --dt 1.0 --dts bbl:0.5,lambda:0.5 --gamma 1.0 \
      --steps 1000 --stride 1000 --save_every 500 --seed $T --run_id a3_estado_T$T
  python -u benchmark/heterogeneous_ensemble.py --engine torchmd \
      --initial_state results/sistemas_multiples/a3_estado_T$T/initial_state.npz \
      --composition cln:8,trp:8,bbl:4,lambda:4 --nnp lambda --precision float32 \
      --temp $T --dt 1.0 --dts bbl:0.5,lambda:0.5 --gamma 1.0 \
      --steps 300000 --stride 1000 --save_every 100 --seed $T \
      --run_id a3_escalera_torchmd_T$T
done
```

```bash
python benchmark/fig_a3_paper.py --discard_ps 50 --plateau_ps 10 25 50 \
    --outdir results/sistemas_multiples/figures
```

The timings of the A3 runs are not a benchmark: the TorchMD runs shared the GPU.
The paper takes timings only from B1.

### Fig. 7: timing

```bash
python benchmark/bench_heterogeneous_ensemble.py --nnp lambda --precision float32 --seed 0
python benchmark/fig_b1_paper.py --b1 results/b1 --outdir results/b1/figures
```

Defaults: M = 4…512 systems (M/4 per protein), 20 warm-up steps, five
synchronized 100-step blocks per mode. `--check` runs a 10-step smoke test.

## Raw outputs

The raw outputs behind Figs. 3–7 are archived on Zenodo, one folder per figure
(**DOI: TODO**). Unpack the archive here, in `paper/`, and rebuild the figures
without rerunning the simulations:

```bash
python benchmark/fig_a2_paper.py --fuerzas fiumd-v0.1.0-results/fig1_forces \
    --nve fiumd-v0.1.0-results/fig2_nve --outdir figures
python benchmark/fig_a3_paper.py --base fiumd-v0.1.0-results/fig3_temperature \
    --discard_ps 50 --plateau_ps 10 25 50 --outdir figures
python benchmark/fig_b1_paper.py --b1 fiumd-v0.1.0-results/fig4_timing --outdir figures
```
