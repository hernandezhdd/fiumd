import torch
import torch.nn as nn

class NeighbourListBuilder(nn.Module):
    def __init__(self, n_pairs, cutoff, dtype=torch.float32, device="cpu"):
        super().__init__()

        self.cutoff = cutoff
        self.cutoff2 = cutoff ** 2
        self.device = device
        self.dtype = dtype
        self.n_pairs = n_pairs

        # Boolean mask of static pairs (set once)
        self.register_buffer("is_static", torch.zeros(n_pairs, dtype=torch.bool, device=device))

        # Active pair indices (static first, then valid neighbours)
        self.register_buffer("active_idx", torch.zeros(n_pairs, dtype=torch.long, device=device))

        # Which of the active_idx are neighbors (points into active_idx)
        self.register_buffer("neigh_map", torch.zeros(n_pairs, dtype=torch.long, device=device))

        # Count of active neighbor pairs (len of neigh_map[:n_nb])
        self.register_buffer("n_nb", torch.zeros(1, dtype=torch.long, device=device))

        # Number of static pairs (set once)
        self.register_buffer("n_static", torch.zeros(1, dtype=torch.long, device=device))

    def set_static_pairs(self, static_pair_indices):
        # static_pair_indices: (n_static,) tensor of global pair indices
        n_static = static_pair_indices.shape[0]
        self.active_idx[:n_static].copy_(static_pair_indices)
        self.is_static[static_pair_indices] = True
        self.n_static.fill_(n_static)

    def update(self, pair_distances):
        # pair_distances: (n_pairs,) (updated before this call)
        with torch.no_grad():
            dynamic_mask = (~self.is_static) & (pair_distances < self.cutoff)
            dynamic_idx = torch.nonzero(dynamic_mask, as_tuple=False).squeeze(1)
            n_nb = dynamic_idx.shape[0]

            # Copy valid dynamic pair indices after statics in active_idx
            self.active_idx[self.n_static.item():self.n_static.item() + n_nb].copy_(dynamic_idx)
            self.n_nb.fill_(n_nb)

            # Fill map from 0 to n_nb (used by forces to index into active_idx)
            self.neigh_map[:n_nb].copy_(torch.arange(n_nb, device=self.device))

    def get_active_indices(self):
        return self.active_idx[:self.n_static.item() + self.n_nb.item()]

    def get_dynamic_indices(self):
        return self.active_idx[self.neigh_map[:self.n_nb.item()]]


def discretize_box(box, subcell_size):
    xbins = torch.arange(0, box[0, 0] + subcell_size, subcell_size)
    ybins = torch.arange(0, box[1, 1] + subcell_size, subcell_size)
    zbins = torch.arange(0, box[2, 2] + subcell_size, subcell_size)
    nxbins = len(xbins) - 1
    nybins = len(ybins) - 1
    nzbins = len(zbins) - 1

    r = torch.tensor([-1, 0, 1])
    neighbour_mask = torch.cartesian_prod(r, r, r)

    cellidx = torch.cartesian_prod(
        torch.arange(nxbins), torch.arange(nybins), torch.arange(nzbins)
    )
    cellneighbours = cellidx.unsqueeze(2) + neighbour_mask.T.unsqueeze(0).repeat(
        cellidx.shape[0], 1, 1
    )

    # Can probably be done easier as we only need to handle -1 and max cases, not general -2, max+1 etc
    nbins = torch.tensor([nxbins, nybins, nzbins])[None, :, None].repeat(
        cellidx.shape[0], 1, 27
    )
    negvals = cellneighbours < 0
    cellneighbours[negvals] += nbins[negvals]
    largevals = cellneighbours > (nbins - 1)
    cellneighbours[largevals] -= nbins[largevals]

    return xbins, ybins, zbins, cellneighbours


# def neighbour_list(pos, box, subcell_size):
#     nsystems = coordinates.shape[0]

#     for s in range(nsystems):
#         spos = pos[s]
#         sbox = box[s]

#         xbins, ybins, zbins = discretize_box(sbox, subcell_size)

#         xidx = torch.bucketize(spos[:, 0], xbins, out_int32=True)
#         yidx = torch.bucketize(spos[:, 1], ybins, out_int32=True)
#         zidx = torch.bucketize(spos[:, 2], zbins, out_int32=True)

#         binidx = torch.stack((xidx, yidx, zidx)).T
