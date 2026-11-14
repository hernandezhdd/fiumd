# Inputs

One folder per protein: `cln` (chignolin), `trp` (Trp-cage), `bbl`, `lambda`
(lambda-repressor). The coarse-grained models, the starting structures and the
trained networks come from the protein-thermodynamics dataset of Majewski et al.,
*Nat. Commun.* 14, 5739 (2023):

- <https://github.com/torchmd/torchmd-protein-thermodynamics> (topologies, prior, starting clusters)
- <http://pub.htmd.org/protein_thermodynamics_data> (embeddings, training coordinates, `Models.zip`)

Only the topologies (`*.psf`) and the experimental reference structures are in
git. Get the rest with, from `paper/`:

```bash
python datos/download_data.py
python benchmark/check_benchmark_data.py --fix --deep
```

The second command writes `model_modified.ckpt` (the old checkpoints keep the
reduction in `reduce_op`, but TorchMD-Net 3 reads `aggr`) and checks that every
file opens.

## Files read by the benchmark

| File (`<m>` = cln, trp, bbl, lambda; `<long>` = chignolin, trpcage, bbl, lambda) | Source |
|---|---|
| `<m>.psf` | `Tutorials/topology/`, **in git** with the bead masses changed from 0.0 to 12.0 |
| `<m>_prior_force_field.yaml` | `Tutorials/prior_force_field.yaml`, the same file for all four |
| `<m>_kcenters_32clusters_coords.xtc` | `Tutorials/starting_xtc/`, source of the initial states |
| `<long>_ca_embeddings.npy` | `training_data/`; loaded but not used (bead IDs come from the atom types) |
| `<long>_5120_ca_coords.npy` | frames of `training_data/<long>_ca_coords.npy`; loaded, used only as a fallback |
| `model.ckpt`, `model_modified.ckpt` | `Models.zip` → `Models/<m>/model.ckpt`; only lambda's is used (`--nnp lambda`) |

**Masses.** The upstream `.psf` files give every bead mass 0.0. TorchMD takes
the masses from the topology before the YAML file, so the initial velocities
come out infinite.

**Coordinate pools.** `download_data.py` reads only the frames it needs, with
HTTP Range requests, and keeps the selection used for the paper:
`np.arange(0, n, n // k)` for cln and trp, sorted `np.random.seed(42)` +
`choice(n, k, replace=False)` for bbl (hence 5076 positions), and
`np.linspace(0, n - 1, k)` for lambda, with `k = 5120 // n_beads`.

<!-- ## Checksums (MD5)

```
3aec29814e86992229eb86f17c0b9aa3  */<m>_prior_force_field.yaml   (all four)
0b7049b58da95a5e2ab277285f163605  cln/cln_kcenters_32clusters_coords.xtc
938d20f79087431bcd65b4b695279e1a  trp/trp_kcenters_32clusters_coords.xtc
75cd0125e6cecc99072104827a20fba3  bbl/bbl_kcenters_32clusters_coords.xtc
85f666a0a3b4856f463f7c289bb89ea9  lambda/lambda_kcenters_32clusters_coords.xtc
5cc21968e3aa57dde24135573e14d312  cln/chignolin_ca_embeddings.npy
5dc99e92fa6cca00735c9e3681a5192f  trp/trpcage_ca_embeddings.npy
85cb6b6c263c1a62764b0b6543665fed  bbl/bbl_ca_embeddings.npy
14bbc454996ebfa45999eef548b55661  lambda/lambda_ca_embeddings.npy
532e1db4414641fc30020a2313b34d0c  cln/chignolin_5120_ca_coords.npy
e4f29ca15dbd00c5c28bcad30cd4d6a9  trp/trpcage_5120_ca_coords.npy
5e1be5bd38828cb34edc7c6762c43955  bbl/bbl_5120_ca_coords.npy
68083ed3102f4ef3844dc798a6e00c8e  lambda/lambda_5120_ca_coords.npy
886367e4fca35f830bc5132e54c9b54a  cln/model.ckpt
818d9557f8eaa77823230baa9aba7200  trp/model.ckpt
2aace54326c8cc3d60f45fe58c705600  bbl/model.ckpt
9dd40cdfef63ca3db18a90851767bbfe  lambda/model.ckpt
8b4969f23685afd8f2dd49ba30698383  lambda/model_modified.ckpt
```

The Trp-cage embeddings used in the paper had K, L and R renumbered
(MD5 `5706566548fd3b48a3c6be3af41ef3e0`). The benchmark never reads their
values, so this does not change any result. -->