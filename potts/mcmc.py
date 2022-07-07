import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from abc import ABC, abstractmethod


class ProposalDistribution(ABC):

    @abstractmethod
    def sample(self, X):
        return


class RandomProposalDistribution(ProposalDistribution, nn.Module):
    '''
    Input is size (batch_size, length)
    X[i, j] is an integer between 0 and K
    '''

    def __init__(self, bs, L, A, device):
        super(RandomProposalDistribution, self).__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device
        self.pos_idx_to_onehot = nn.Embedding(L, L)
        self.pos_idx_to_onehot.weight = nn.Parameter(torch.eye(L),
                                                     requires_grad=False)

    def sample(self, X):
        bs, L, A, device = self.bs, self.L, self.A, self.device
        positions_to_mutate = self.pos_idx_to_onehot(
            torch.randint(L, size=(bs, ), device=device))
        # low=1 enforces that a mutation is made
        mutations = torch.randint(low=1, high=A, size=(bs, L), device=device)
        new_state = (X + (mutations * positions_to_mutate)) % A
        return new_state.long()


class ReplayBuffer(object):

    def __init__(self,
                 buffer_size,
                 batch_size,
                 length,
                 num_cats,
                 reinit_freq=0.05):
        super(ReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length = length
        self.num_cats = num_cats
        self.reinit_freq = reinit_freq
        self.buffer = torch.randint(high=num_cats, size=(buffer_size, length))
        if num_cats == 2:
            self.buffer = self.buffer.to(torch.float)
        self.inds = None

    def sample(self):
        inds = torch.randint(self.buffer_size, size=(self.batch_size, ))
        buffer_samples = self.buffer[inds]
        random_samples = torch.randint(high=self.num_cats,
                                       size=(self.batch_size, self.length))
        choose_random = (torch.rand(self.batch_size) <
                         self.reinit_freq).int()[:, None]
        samples = choose_random * random_samples + (
            1 - choose_random) * buffer_samples
        self.inds = inds
        return samples

    def update(self, samples):
        self.buffer[self.inds] = samples.cpu()


###################################################################################################
class CategoricalMetropolistHastingsSampler(ABC):

    def sample(self, X, model):
        model.eval()
        bs, L, A, device = self.bs, self.L, self.A, self.device
        # get negative energy and proposal distribution
        f_x, forward_proposal = self.compute_neg_energy_and_proposal(X, model)
        # sample from proposal
        pos_aa_f = forward_proposal.sample()
        pos_f = (pos_aa_f // A)
        aa_f = (pos_aa_f - (pos_f * A))
        # generate full samples from position + category
        X_f = X.scatter(dim=1,
                        index=pos_f.unsqueeze(-1),
                        src=aa_f.unsqueeze(-1))
        # get original category
        aa_r = X.gather(dim=1, index=pos_f.unsqueeze(-1)).squeeze()
        pos_aa_r = (pos_f * A + aa_r)
        f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
            X_f, model)
        forward_log_p = forward_proposal.log_prob(pos_aa_f)
        reverse_log_p = reverse_proposal.log_prob(pos_aa_r)
        log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
        log_unifs = torch.log(torch.rand(bs, device=device))
        accept = (log_alpha >= log_unifs).long()[:, None]
        samples = (accept * X_f) + ((1 - accept) * X)
        model.train()
        return samples, accept

    def sample_nsteps(self, X, model, n_steps):
        model.eval()
        bs, L, A, device = self.bs, self.L, self.A, self.device
        # get negative energy and proposal distribution
        f_x, forward_proposal = self.compute_neg_energy_and_proposal(X, model)
        acc_rate = 0
        per_change = 0
        X_orig = X.clone()
        for step in range(n_steps):
            # sample from proposal
            pos_aa_f = forward_proposal.sample()
            pos_f = (pos_aa_f // A)
            aa_f = (pos_aa_f - (pos_f * A))
            # generate full samples from position + category
            X_f = X.scatter(dim=1,
                            index=pos_f.unsqueeze(-1),
                            src=aa_f.unsqueeze(-1))
            # get original category
            aa_r = X.gather(dim=1, index=pos_f.unsqueeze(-1)).squeeze()
            pos_aa_r = (pos_f * A + aa_r)
            f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
                X_f, model)
            forward_log_p = forward_proposal.log_prob(pos_aa_f)
            reverse_log_p = reverse_proposal.log_prob(pos_aa_r)
            log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
            log_unifs = torch.log(torch.rand(bs, device=device))
            accept = (log_alpha >= log_unifs).long()[:, None]
            X_f = (accept * X_f) + ((1 - accept) * X)
            per_change += ((X_f != X).sum(dim=1).sum() / X.shape[0]).item()
            X = X_f
            probs = (accept * reverse_proposal.probs) + (
                (1 - accept) * forward_proposal.probs)
            forward_proposal = Categorical(probs)
            accept = accept.squeeze(-1)
            f_x = (accept * f_x_f) + ((1 - accept) * f_x)
            acc_rate += accept.sum() / accept.shape[0]

        model.train()
        print(
            f"Avg Acc: {acc_rate / n_steps:.2f}, Actual Acc: {per_change / n_steps:.2f}"
        )
        return X

    @abstractmethod
    def compute_neg_energy_and_proposal(self, X, model):
        return


class UniformCategoricalSampler(CategoricalMetropolistHastingsSampler):
    '''
    Input is size (batch_size, length)
    X[i, j] is an integer between 0 and K
    '''

    def __init__(self, bs, L, A, device):
        '''
        bs: batch size
        L: number of positions
        A: number of categories
        '''
        super(UniformCategoricalSampler, self).__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        f_x = -model(X)
        bs, L, A, device = self.bs, self.L, self.A, self.device
        probs = torch.ones(bs, L * A, device=device) * 1 / (L * A)
        dist = Categorical(probs)
        return f_x, dist


class GWGCategoricalSampler(CategoricalMetropolistHastingsSampler):
    '''
    Input is size (batch_size, length)
    X[i, j] is an integer between 0 and K
    '''

    def __init__(self, bs, L, A, device):
        '''
        bs: batch size
        L: number of positions
        A: number of categories
        '''
        super(GWGCategoricalSampler, self).__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device
        to_onehot = nn.Embedding(A, A).to(device)
        to_onehot.weight = nn.Parameter(torch.eye(A, device=device),
                                        requires_grad=False)
        self.to_onehot = to_onehot

    def compute_neg_energy_and_proposal(self, X, model):
        bs, L, A, device = self.bs, self.L, self.A, self.device
        X_onehot = self.to_onehot(X)
        X_onehot.requires_grad_()
        f_x = -model.energy_from_onehot(X_onehot)
        grad_f_x = torch.autograd.grad(f_x.sum(), X_onehot,
                                       retain_graph=True)[0]
        with torch.no_grad():
            d_tilde = (grad_f_x -
                       (X_onehot * grad_f_x).sum(dim=-1).unsqueeze(dim=-1))
        probs = torch.softmax(d_tilde.reshape(d_tilde.shape[0], -1) / 2,
                              dim=-1)
        dist = Categorical(probs)
        return f_x, dist


#################################################################################################


class BinaryMetropolistHastingsSampler(ABC):

    def sample(self, X, model):
        model.eval()
        bs, device = self.bs, self.device
        # get negative energy and proposal distribution
        f_x, forward_proposal = self.compute_neg_energy_and_proposal(X, model)
        # sample from proposal
        pos_f = forward_proposal.sample()
        # generate full samples from position + category
        flip_bit = (X.gather(dim=1, index=pos_f.unsqueeze(-1)) + 1) % 2
        X_f = X.scatter(dim=1, index=pos_f.unsqueeze(-1), src=flip_bit)
        f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
            X_f, model)
        forward_log_p = forward_proposal.log_prob(pos_f)
        reverse_log_p = reverse_proposal.log_prob(pos_f)
        log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
        log_unifs = torch.log(torch.rand(bs, device=device))
        accept = (log_alpha >= log_unifs).long()[:, None]
        samples = (accept * X_f) + ((1 - accept) * X)
        model.train()
        return samples, accept

    def sample_nsteps(self, X, model, n_steps, pbar=False):
        model.eval()
        bs, device = self.bs, self.device
        # get negative energy and proposal distribution
        f_x, forward_proposal = self.compute_neg_energy_and_proposal(X, model)
        acc_rate = 0
        if pbar:
            iterator = tqdm(range(n_steps), total=n_steps)
        else:
            iterator = range(n_steps)
        for step in iterator:
            with torch.no_grad():
                pos_f = forward_proposal.sample()
                # generate full samples from position + category
                flip_bit = (X.gather(dim=1, index=pos_f.unsqueeze(-1)) + 1) % 2
                X_f = X.scatter(dim=1, index=pos_f.unsqueeze(-1), src=flip_bit)
            f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
                X_f, model)
            with torch.no_grad():
                forward_log_p = forward_proposal.log_prob(pos_f)
                reverse_log_p = reverse_proposal.log_prob(pos_f)
                log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
                log_unifs = torch.log(torch.rand(bs, device=device))
                accept = (log_alpha >= log_unifs).long()[:, None]
                X = (accept * X_f) + ((1 - accept) * X)
                probs = (accept * reverse_proposal.probs) + (
                    (1 - accept) * forward_proposal.probs)
                forward_proposal = Categorical(probs)
                accept = accept.squeeze(-1)
                f_x = (accept * f_x_f) + ((1 - accept) * f_x)
                acc_rate += accept.sum() / accept.shape[0]
        model.train()
        print(f"Avg Acc: {acc_rate / n_steps:.2f}")
        return X

    @abstractmethod
    def compute_neg_energy_and_proposal(self, X, model):
        return


class UniformBinarySampler(BinaryMetropolistHastingsSampler):
    '''
    Input is size (batch_size, length)
    X[i, j] is an integer between 0 and 1
    '''

    def __init__(self, bs, L, device):
        '''
        bs: batch size
        L: number of positions
        '''
        super(UniformBinarySampler, self).__init__()
        self.bs = bs
        self.L = L
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        f_x = -model(X)
        bs, L, device = self.bs, self.L, self.device
        probs = torch.ones(bs, L, device=device) * 1 / (L)
        dist = Categorical(probs)
        return f_x, dist


class GWGBinarySampler(BinaryMetropolistHastingsSampler):
    '''
    Input is size (batch_size, length)
    X[i, j] is an integer between 0 and 1
    '''

    def __init__(self, bs, L, device):
        '''
        bs: batch size
        L: number of positions
        '''
        super(GWGBinarySampler, self).__init__()
        self.bs = bs
        self.L = L
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        X.requires_grad_()
        f_x = -model(X)
        grad_f_x = torch.autograd.grad(f_x.sum(), X, retain_graph=False)[0]
        with torch.no_grad():
            d_tilde = -(2 * X - 1) * grad_f_x
        probs = torch.softmax(d_tilde / 2, dim=-1)
        dist = Categorical(probs)
        return f_x, dist
