import time
import torch
import numpy as np
from itertools import combinations, product
from .mcmc import GWGCategoricalSampler, ReplayBuffer, CategoricalGibbsSampler, OneHotCategorical, CategoricalMetropolistHastingsSampler, PottsGWGCategoricalSampler, PottsGWGHardWallCategoricalSampler
from .potts import Potts
from .potts import get_subset_potts_optimized
from collections import Counter
from tempfile import NamedTemporaryFile
import subprocess
from queue import PriorityQueue
from joblib import Parallel, delayed, parallel_backend


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
            crossovers = list(crossovers)
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


def get_reduced_potts(model, rl, block_to_update, seq_idx_to_update, device,
                      entropy_reg):
    L = model.L
    A = model.A
    h = model.h.reshape((L, A)).detach().cpu().numpy().copy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()

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

    sub_model = Potts(h=new_h, W=sub_W)
    sub_model = sub_model.to(device)
    return sub_model


def potts_to_pgmpy(model):
    from pgmpy.models import MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor
    from pgmpy.inference import Mplp
    L = model.L
    A = model.A
    h = model.h.reshape((L, A)).detach().cpu().numpy().copy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()

    mm = MarkovNetwork()
    mm.add_nodes_from([f"x_{i}" for i in range(L)])
    mm.add_edges_from([(f"x_{i}", f"x_{j}") for i in range(L - 1)
                       for j in range(i + 1, L)])
    factors = []
    for i, node in enumerate(mm.nodes):
        factor = DiscreteFactor([node], [A], -h[i])
        factors.append(factor)
    for i in range(L - 1):
        node_i = f"x_{i}"
        for j in range(i + 1, L):
            node_j = f"x_{j}"
            factor = DiscreteFactor([node_i, node_j], [A, A],
                                    -W[i, j].reshape(-1))
            factors.append(factor)
    mm.add_factors(*factors)
    return mm


def potts_to_LG_file(model):
    L = model.L
    A = model.A
    h = model.h.reshape((L, A)).detach().cpu().numpy().copy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()

    temp_fh = NamedTemporaryFile(suffix=".LG")
    temp_fn = temp_fh.name
    with open(temp_fn, 'w') as outfn:
        outfn.write("MARKOV\n")
        outfn.write(f"{L}\n")
        outfn.write(f" ".join([f'{A}' for _ in range(L)]) + '\n')
        num_functions = L + (L * (L - 1) // 2)
        outfn.write(f'{num_functions}\n')
        for i in range(L):
            outfn.write(f"1 {i}\n")
        for i in range(L - 1):
            for j in range(i + 1, L):
                outfn.write(f"2 {i} {j}\n")
        outfn.write("\n")
        for i in range(L):
            outfn.write(f"{A}\n")
            outfn.write(" ".join([f'{x:.7f}' for x in -h[i]]) + "\n")
        for i in range(L - 1):
            for j in range(i + 1, L):
                outfn.write(f"{A**2}\n")
                outfn.write(
                    " ".join([f'{x:.7f}'
                              for x in -W[i, j].reshape(-1)]) + "\n")
    return temp_fh


def run_toulbar2(lg_file, toulbar2_path='toulbar2'):
    cmd = f"{toulbar2_path} -s 1 {lg_file.name}"
    result = subprocess.run(cmd,
                            check=True,
                            shell=True,
                            cwd="/tmp",
                            stdout=subprocess.PIPE)
    lines = result.stdout.decode("utf-8").strip().split("\n")
    solution_idxs = [i for i, l in enumerate(lines) if "New solution" in l]
    map_soln = [int(x) for x in lines[solution_idxs[-1] + 1].split()]
    return map_soln


def run_greedy_toulbar2(lg_file, num_solns=100, toulbar2_path='toulbar2'):
    cmd = f"{toulbar2_path} -s 1 -a={num_solns} {lg_file.name}"
    result = subprocess.run(cmd,
                            check=True,
                            shell=True,
                            stdout=subprocess.PIPE)
    lines = result.stdout.decode("utf-8").strip().split("\n")
    solns = [list(map(int, l.split()[2:])) for l in lines if "solution(" in l]
    return solns


def coordinate_descent_anneal(model,
                              rl,
                              block_to_update,
                              seq_idx_to_update,
                              device='cpu',
                              entropy_reg=0.005):

    block_seqs = rl.block_to_seqs[block_to_update]
    seq_to_update = block_seqs[seq_idx_to_update]
    other_seqs = np.asarray(
        [s for i, s in enumerate(block_seqs) if i != seq_idx_to_update])

    sub_model = get_reduced_potts(model, rl, block_to_update,
                                  seq_idx_to_update, device, entropy_reg)
    seq_to_update = torch.tensor(seq_to_update)
    energy_before = sub_model.to('cpu')(torch.nn.functional.one_hot(
        seq_to_update,
        num_classes=sub_model.A).unsqueeze(0).to(torch.float)).item()
    new_seq = anneal(sub_model, seq_to_update, device=device)
    energy_after = sub_model.to('cpu')(torch.nn.functional.one_hot(
        new_seq, num_classes=sub_model.A).unsqueeze(0).to(torch.float)).item()
    if energy_after >= energy_before:
        new_seq = seq_to_update.clone()
    return new_seq.tolist(), new_seq.tolist() == seq_to_update.tolist()


def coordinate_descent_toulbar2(model,
                                rl,
                                block_to_update,
                                seq_idx_to_update,
                                device='cpu',
                                entropy_reg=0.005):

    block_seqs = rl.block_to_seqs[block_to_update]
    seq_to_update = block_seqs[seq_idx_to_update]
    other_seqs = np.asarray(
        [s for i, s in enumerate(block_seqs) if i != seq_idx_to_update])

    sub_model = get_reduced_potts(model, rl, block_to_update,
                                  seq_idx_to_update, device, entropy_reg)
    lg_file = potts_to_LG_file(sub_model)
    new_seq = run_toulbar2(lg_file)

    old_energy = sub_model(
        torch.nn.functional.one_hot(torch.tensor(new_seq),
                                    num_classes=model.A).unsqueeze(0).to(
                                        torch.float)).item()

    new_energy = sub_model(
        torch.nn.functional.one_hot(torch.tensor(new_seq),
                                    num_classes=model.A).unsqueeze(0).to(
                                        torch.float)).item()
    assert new_energy <= old_energy
    b_and_b = BranchAndBoundToulbar2(sub_model)
    while np.any(np.all(np.asarray(new_seq) == other_seqs, axis=1)):
        energy, new_seq = next(b_and_b)
    return new_seq, new_seq == seq_to_update.tolist()


def coordinate_descent_mplp(model,
                            rl,
                            block_to_update,
                            seq_idx_to_update,
                            device='cuda',
                            entropy_reg=0.005):
    from pgmpy.inference import Mplp
    sub_model = get_reduced_potts(model, rl, block_to_update,
                                  seq_idx_to_update, device, entropy_reg)
    pgmpy_potts = potts_to_pgmpy(sub_model)
    inference = Mplp(pgmpy_potts)
    res = inference.map_query()
    map_state = list(res[f'x_{i}'] for i in range(len(res)))
    return map_state


def simulated_annealing(model,
                        rl,
                        block_to_update,
                        seq_idx_to_update,
                        bs=1000,
                        n_steps=500,
                        device='cuda',
                        entropy_reg=0.005,
                        temp=1.0):
    sub_model = get_reduced_potts(model, rl, block_to_update,
                                  seq_idx_to_update, device, entropy_reg)
    sub_model.temp = temp
    sub_L = sub_model.L
    A = sub_model.A
    #samples = torch.randint(0, A, size=(bs, sub_L))
    # add sequence to samples
    seq_to_update = rl.block_to_seqs[block_to_update][seq_idx_to_update]
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
        if within_block_pairs:
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


def anneal(model,
           seq,
           num_chains=1000,
           num_temp_steps=10,
           n_steps=20,
           device='cuda',
           wt_onehot=None,
           max_mutations=-1):
    L = model.L
    A = model.A
    model.to(device)
    samples = seq.unsqueeze(0).repeat(num_chains, 1).to(device)
    samples = torch.nn.functional.one_hot(samples, num_classes=A).to(
        torch.float).to(device)
    energies_before = model(samples)
    temp = 1.0
    for _ in range(num_temp_steps):
        model.temp = temp
        if max_mutations != -1:
            assert wt_onehot is not None
            sampler = PottsGWGHardWallCategoricalSampler(
                num_chains,
                L,
                A,
                device,
                wt_onehot=wt_onehot,
                max_mutations=max_mutations)
        else:
            sampler = PottsGWGCategoricalSampler(num_chains, L, A, device)
        samples = sampler.sample_nsteps(samples, model, n_steps)
        energies_after = model(samples)
        temp = 0.9 * temp
        samples = samples.to('cpu')
        copy_of_best = samples[energies_after.argmin().item()].cpu().argmax(
            dim=-1).unsqueeze(0).repeat(num_chains, 1)
        samples = torch.nn.functional.one_hot(copy_of_best,
                                              num_classes=A).to(torch.float)
        samples = samples.to(device)
        #num_mutations = (L - (samples.reshape((-1, L*A)) @ wt_onehot.to(device).reshape((-1, L*A)).T).reshape(-1))
    return samples[0].cpu().argmax(dim=-1)


class BranchAndBoundToulbar2():

    def __init__(self, model):
        self.model = model.cpu()
        self.L = model.L
        self.A = model.A
        self.pq = PriorityQueue()
        self.h = model.h.reshape(
            (self.L, self.A)).detach().cpu().numpy().copy()
        self.W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()
        self.device = "cpu"
        # leaf node
        self.pq.put((np.inf, []))

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            energy, node = self.pq.get()
            if self.reached_leaf_node(node):
                return energy, node
            else:
                self.expand_node(node)

    def expand_node(self, node):
        new_nodes = []
        for i in range(self.A):
            new_node = node.copy()
            new_node.append(i)
            new_nodes.append(new_node)

        # multiprocessing cannot use queues
        all_items = []
        while not self.pq.empty():
            all_items.append(self.pq.get())
        self.pq = None
        # scores = Parallel(n_jobs=-1,
        #                   prefer='threads')(delayed(self.score_node)(x)
        #                                     for x in new_nodes)
        scores = [self.score_node(x) for x in new_nodes]
        self.pq = PriorityQueue()
        for x in all_items:
            self.pq.put(x)
        for score, new_node in zip(scores, new_nodes):
            self.pq.put((score, new_node))

    def score_node(self, node):
        if self.reached_leaf_node(node):
            total_state = node
            with torch.no_grad():
                X = torch.tensor(total_state)
                X = torch.nn.functional.one_hot(X, num_classes=self.A).to(
                    torch.float).to(self.device)
                energy = self.model(X.unsqueeze(0))[0].cpu().item()
                return energy
        else:
            # score node by solving ILP
            sub_model = self.generate_reduced_model(node)
            lg_file = potts_to_LG_file(sub_model)
            map_state = run_toulbar2(lg_file)
            total_state = node + map_state
            lg_file.close()
            with torch.no_grad():
                X = torch.tensor(total_state)
                X = torch.nn.functional.one_hot(X, num_classes=self.A).to(
                    torch.float).to(self.device)
                energy = self.model(X.unsqueeze(0))[0].cpu().item()
                return energy

    def generate_reduced_model(self, node):
        h = self.h.copy()
        W = self.W.copy()
        A = self.A
        L = self.L

        num_pos_fixed = len(node)
        sub_L = L - num_pos_fixed
        assert (sub_L > 0)
        remaining_positions = [i for i in range(num_pos_fixed, L)]
        assert (len(remaining_positions) > 0)

        if False:
            sub_h = h[remaining_positions]
            sub_W = W[remaining_positions][:, remaining_positions]
            sub_W = sub_W.transpose((0, 2, 1, 3)).reshape(sub_L * A, sub_L * A)

            h_tilde = np.zeros((sub_L, A))
            for pos_i in range(num_pos_fixed):
                aa_i = node[pos_i]
                for pos_j in range(num_pos_fixed, L):
                    for aa_j in range(A):
                        h_tilde[pos_j - num_pos_fixed, aa_j] += W[pos_i, pos_j,
                                                                  aa_i, aa_j]
            sub_h += h_tilde
            new_model = Potts(h=torch.tensor(sub_h), W=torch.tensor(sub_W))
            return new_model
        else:
            sub_h, sub_W = get_subset_potts_optimized(torch.tensor(h), torch.tensor(W).transpose(1,2).reshape(L*A, L*A), np.asarray(remaining_positions), np.asarray(node + [0 for _ in range(L - len(node))]))
            subset_model = Potts(sub_h.shape[0], 21)
            subset_model.load_from_weights(h=sub_h, W=sub_W)
            return subset_model



    def reached_leaf_node(self, node):
        return len(node) == self.L


class BranchAndBoundMplp():
    def __init__(self, model):
        self.model = model.cpu()
        self.L = model.L
        self.A = model.A
        self.pq = PriorityQueue()
        self.h = model.h.reshape(
            (self.L, self.A)).detach().cpu().numpy().copy()
        self.W = model.reshape_to_L_L_A_A().detach().cpu().numpy().copy()
        self.device = "cpu"
        # leaf node
        self.pq.put((np.inf, []))

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            energy, node = self.pq.get()
            if self.reached_leaf_node(node):
                return energy, node
            else:
                self.expand_node(node)

    def expand_node(self, node):
        new_nodes = []
        for i in range(self.A):
            new_node = node.copy()
            new_node.append(i)
            new_nodes.append(new_node)

        # multiprocessing cannot use queues
        all_items = []
        while not self.pq.empty():
            all_items.append(self.pq.get())
        self.pq = None
        if False:
            scores = Parallel(n_jobs=-1,
                              prefer='threads')(delayed(self.score_node)(x)
                                                for x in new_nodes)
        else:
            scores = [self.score_node(x) for x in new_nodes]
        self.pq = PriorityQueue()
        for x in all_items:
            self.pq.put(x)
        for score, new_node in zip(scores, new_nodes):
            self.pq.put((score, new_node))

    def score_node(self, node):
        from pgmpy.inference import Mplp
        if self.reached_leaf_node(node):
            total_state = node
            with torch.no_grad():
                X = torch.tensor(total_state)
                X = torch.nn.functional.one_hot(X, num_classes=self.A).to(
                    torch.float).to(self.device)
                energy = self.model(X.unsqueeze(0))[0].cpu().item()
                return energy
        else:
            # score node by solving ILP
            sub_model = self.generate_reduced_model(node)
            pgmpy_potts = potts_to_pgmpy(sub_model)
            inference = Mplp(pgmpy_potts)
            map_d = inference.map_query()
            map_state = [map_d[f"x_{i}"] for i in range(len(map_d))]
            total_state = node + map_state
            with torch.no_grad():
                X = torch.tensor(total_state)
                X = torch.nn.functional.one_hot(X, num_classes=self.A).to(
                    torch.float).to(self.device)
                energy = self.model(X.unsqueeze(0))[0].cpu().item()
                return energy

    def generate_reduced_model(self, node):
        h = self.h.copy()
        W = self.W.copy()
        A = self.A
        L = self.L
        if False:
            num_pos_fixed = len(node)
            sub_L = L - num_pos_fixed
            assert (sub_L > 0)
            remaining_positions = [i for i in range(num_pos_fixed, L)]
            assert (len(remaining_positions) > 0)
            sub_h = h[remaining_positions]
            sub_W = W[remaining_positions][:, remaining_positions]
            sub_W = sub_W.transpose((0, 2, 1, 3)).reshape(sub_L * A, sub_L * A)

            h_tilde = np.zeros((sub_L, A))
            for pos_i in range(num_pos_fixed):
                aa_i = node[pos_i]
                for pos_j in range(num_pos_fixed, L):
                    for aa_j in range(A):
                        h_tilde[pos_j - num_pos_fixed, aa_j] += W[pos_i, pos_j,
                                                                  aa_i, aa_j]
            sub_h += h_tilde
            new_model = Potts(h=torch.tensor(sub_h) - (1e-6),
                              W=torch.tensor(sub_W) - (1e-6))
            return new_model
        else:
            sub_h, sub_W = get_subset_potts_optimized(torch.tensor(h), torch.tensor(W).transpose(1,2).reshape(L*A, L*A), np.asarray(remaining_positions), np.asarray(node + [0 for _ in range(L - len(node))]))
            subset_model = Potts(sub_h.shape[0], 21)
            subset_model.load_from_weights(h=sub_h, W=sub_W)
            return subset_model

    def reached_leaf_node(self, node):
        return len(node) == self.L
