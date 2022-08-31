import torch
import numpy as np
from itertools import combinations, product
from .mcmc import GWGCategoricalSampler, ReplayBuffer, CategoricalGibbsSampler, OneHotCategorical, CategoricalMetropolistHastingsSampler, PottsGWGCategoricalSampler
from .potts import Potts
from collections import Counter


class RecombinationLibrary():

    def __init__(self,
                 crossovers,
                 seq_len,
                 parents=None,
                 n_parents=None,
                 A=21):
        super().__init__()
        assert (parents is not None) or (
            n_parents
            is not None), "Either parents or n_parents must be provided"
        assert crossovers[0] == 0, "First crossover must be 0"
        self.L = seq_len
        self.A = A
        assert crossovers[-1] <= seq_len
        if crossovers[-1] < seq_len:
            crossovers.append(seq_len)

        self.crossovers = crossovers

        if parents is None:
            self.n_parents = n_parents
        else:
            self.n_parents = parents.shape[0]
        self.n_blocks = len(self.crossovers) - 1
        self.block_to_seqs = {}
        self.block_to_pos = {}
        for i in range(self.n_blocks):
            start = self.crossovers[i]
            end = self.crossovers[i + 1]
            # make n_parents a parameter
            if parents is None:
                seqs = np.random.randint(0,
                                         high=A,
                                         size=(n_parents,
                                               end - start)).tolist()
            else:
                seqs = parents[:, start:end]
            self.block_to_seqs[i] = seqs
            self.block_to_pos[i] = np.arange(start, end)

    def compute_h_tilde(self,
                        block_to_update,
                        seq_idx_to_update,
                        W,
                        entropy_reg=0.005):
        assert (0 <= block_to_update) and (block_to_update < self.n_blocks)
        assert W.shape == (self.L, self.L, self.A, self.A)

        positions_i = self.block_to_pos[block_to_update]
        h_tilde = np.zeros((len(positions_i), self.A))
        # for each position in the block
        for i, pos_i in enumerate(positions_i):
            # for each amino acid in the position
            for aa_i in range(self.A):
                # for all the other blocks
                for block_j in range(self.n_blocks):
                    if block_j != block_to_update:
                        positions_at_block = self.block_to_pos[block_j]
                        # for all the sequences in that block
                        for seq_j in self.block_to_seqs[block_j]:
                            # for all the positions in that sequence
                            for j, pos_j in enumerate(positions_at_block):
                                h_tilde[i, aa_i] += W[pos_i, pos_j, aa_i,
                                                      seq_j[j]]
        h_tilde /= (self.n_parents)

        seqs = self.block_to_seqs[block_to_update]
        other_seqs = np.asarray(
            [s for i, s in enumerate(seqs) if i != seq_idx_to_update])
        for pos in range(len(other_seqs[0])):
            column = other_seqs[:, pos]
            counter = Counter(column)
            for aa in range(self.A):
                h_tilde[pos, aa] += (entropy_reg * counter[aa])

        return h_tilde


def simulated_annealing(model,
                        rl,
                        block_to_update,
                        seq_idx_to_update,
                        bs=1000,
                        n_steps=500,
                        device='cuda',
                        entropy_reg=0.005,
                        temp=1.0):
    L = model.L
    A = model.A
    h = model.h.reshape((L, A)).detach().cpu().numpy().copy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()

    seq_to_update = rl.block_to_seqs[block_to_update][seq_idx_to_update]
    h_tilde = rl.compute_h_tilde(block_to_update,
                                 seq_idx_to_update,
                                 W,
                                 entropy_reg=entropy_reg)
    positions_in_block = rl.block_to_pos[block_to_update]

    sub_L = len(positions_in_block)
    sub_h = h[positions_in_block]
    new_h = sub_h + h_tilde
    sub_W = W[positions_in_block][:, positions_in_block]
    sub_W = sub_W.transpose((0, 2, 1, 3)).reshape(sub_L * A, sub_L * A)

    sub_model = Potts(h=sub_h, W=sub_W)
    sub_model = sub_model.to(device)
    sub_model.temp = temp

    #samples = torch.randint(0, A, size=(bs, sub_L))
    # add sequence to samples
    s = torch.tensor(seq_to_update)
    samples = s.unsqueeze(0).repeat(bs, 1)

    samples = torch.nn.functional.one_hot(samples,
                                          num_classes=A).to(torch.float)
    sampler = PottsGWGCategoricalSampler(bs, sub_L, A, device)
    samples = samples.to(device)
    energies_before = sub_model(samples)
    #print(f"Before: {energies_before.min():.3f}")

    samples = sampler.sample_nsteps(samples, sub_model, n_steps)
    energies = sub_model(samples)
    samples = samples.cpu().detach().argmax(dim=-1).numpy()
    best = samples[energies.argmin()]
    #print(f"After: {energies.min():.3f}")

    return best.tolist()


# Adapted from aav_evasion/raspp_design.ipynb
def avg_energy_by_formula(parents, crossovers, is_coef, p_coef):
    n_parents = parents.shape[0]
    seq_len = parents.shape[1]
    if crossovers[0] <= 0:
        crossovers[0] = 0
    else:
        crossovers = np.concatenate([(0, ), crossovers])
    if crossovers[-1] >= seq_len:
        crossovers[-1] = seq_len
    else:
        crossovers = np.concatenate([crossovers, (seq_len, )])
    E = 0.
    E += np.sum(
        is_coef[np.concatenate([np.arange(seq_len) for _ in range(n_parents)]),
                parents.flatten()]) / n_parents
    for start, end in zip(crossovers[:-1], crossovers[1:]):
        within_block_pairs = list(combinations(range(start, end), 2))
        i, j = zip(*within_block_pairs)
        k, l = parents[:, i].flatten(), parents[:, j].flatten()
        i = np.concatenate([i for _ in range(n_parents)])
        j = np.concatenate([j for _ in range(n_parents)])
        E += np.sum(p_coef[i, j, k, l]) / n_parents

        if seq_len > end:
            between_block_pairs = list(
                product(range(start, end), range(end, seq_len)))
            i, j = zip(*between_block_pairs)
            k = np.concatenate([
                parents[p, i].flatten() for p in range(n_parents)
                for _ in range(n_parents)
            ])
            l = np.concatenate(
                [parents[:, j].flatten() for _ in range(n_parents)])
            i = np.concatenate([i for _ in range(n_parents**2)])
            j = np.concatenate([j for _ in range(n_parents**2)])
            E += np.sum(p_coef[i, j, k, l]) / (n_parents**2)
    return E
