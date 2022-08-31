import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical
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
        super().__init__()
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
        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length = length
        self.num_cats = num_cats
        self.reinit_freq = reinit_freq
        if num_cats > 2:
            dist = OneHotCategorical(logits=torch.ones(size=(buffer_size,
                                                             length,
                                                             num_cats)))
            self.buffer = dist.sample().to(torch.float)
        else:
            self.buffer = torch.randint(high=num_cats,
                                        size=(buffer_size,
                                              length)).to(torch.float)
        assert (self.batch_size <= self.buffer_size)
        self.inds = None

    def sample(self):
        inds = torch.randperm(self.buffer_size)[:self.batch_size]
        buffer_samples = self.buffer[inds]
        if self.num_cats > 2:
            dist = OneHotCategorical(logits=torch.ones(size=(self.batch_size,
                                                             self.length,
                                                             self.num_cats)))
            random_samples = dist.sample().to(torch.float)
            choose_random = (torch.rand(self.batch_size) <
                             self.reinit_freq).int()[:, None, None]
        else:
            random_samples = torch.randint(high=self.num_cats,
                                           size=(self.buffer_size,
                                                 self.length)).to(torch.float)
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
        acc_rate = 0
        per_change = 0
        X_orig = X.clone()
        # sample from proposal
        changes = forward_proposal.sample()

        forward_log_p = forward_proposal.log_prob(changes)
        # reshape to (bs, L, A)
        changes_r = changes.view(X.size())
        # get binary indicator (bs, L) indicating which dim was changed
        changed_ind = changes_r.sum(-1)
        X_f = X.clone() * (1. - changed_ind[:, :, None]) + changes_r
        # get original category
        f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
            X_f, model)
        reverse_changes = X * changed_ind[:, :, None]
        reverse_log_p = reverse_proposal.log_prob(
            reverse_changes.view(X.size(0), -1))
        log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
        log_unifs = torch.log(torch.rand(bs, device=device))
        accept = (log_alpha >= log_unifs).float()[:, None, None]
        X_temp = X_f.clone()
        X_f = (accept * X_f) + ((1 - accept) * X)
        per_change += ((X_f != X).sum(dim=1).sum() / X.shape[0]).item()

        X = X_f.detach()

        model.train()
        return X, accept.squeeze()

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
            changes = forward_proposal.sample()

            forward_log_p = forward_proposal.log_prob(changes)
            # reshape to (bs, L, A)
            changes_r = changes.view(X.size())
            # get binary indicator (bs, L) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            X_f = X.clone() * (1. - changed_ind[:, :, None]) + changes_r
            # get original category
            f_x_f, reverse_proposal = self.compute_neg_energy_and_proposal(
                X_f, model)
            reverse_changes = X * changed_ind[:, :, None]
            reverse_log_p = reverse_proposal.log_prob(
                reverse_changes.view(X.size(0), -1))
            log_alpha = (f_x_f - f_x) + (reverse_log_p - forward_log_p)
            log_unifs = torch.log(torch.rand(bs, device=device))
            accept = (log_alpha >= log_unifs).float()[:, None, None]
            X_temp = X_f.clone()
            X_f = (accept * X_f) + ((1 - accept) * X)
            per_change += ((X_f != X).sum(dim=1).sum() / X.shape[0]).item()

            X = X_f.detach()
            accept = accept.squeeze(-1)

            probs = (accept * reverse_proposal.probs) + (
                (1 - accept) * forward_proposal.probs)
            forward_proposal = OneHotCategorical(probs=probs)

            accept = accept.squeeze(-1)
            f_x = (accept * f_x_f) + ((1 - accept) * f_x)
            acc_rate += accept.sum() / accept.shape[0]
        model.train()
        # print(
        #     f"Avg Acc: {acc_rate / n_steps:.2f}, Actual Acc: {per_change / n_steps:.2f}"
        # )
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
        super().__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        f_x = -model(X)
        L, A, device = self.L, self.A, self.device
        probs = torch.ones(X.size(0), L * A, device=device) * 1 / (L * A)
        dist = OneHotCategorical(probs)
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
        super().__init__()
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
        super().__init__()
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


class CategoricalGibbsSampler(nn.Module):

    def __init__(self, L, A, rand=False):
        super().__init__()
        self.L = L
        self.A = A

        self._i = 0

        self._ar = 0.
        self._hops = 0.
        self.rand = rand

    def sample(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.L)
        else:
            i = self._i

        logits = []
        A = self.A

        for k in range(A):
            sample = x.clone()
            sample_i = torch.zeros((A, ))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = -model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.L
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def sample_nsteps(self, X, model, n_steps):
        for step in range(n_steps):
            X = self.sample(X, model)
        return X


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
        super().__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        bs, L, A, device = self.bs, self.L, self.A, self.device
        X.requires_grad_()
        f_x = -model(X)
        grad_f_x = torch.autograd.grad(f_x.sum(), X, retain_graph=True)[0]

        with torch.no_grad():
            d_tilde = (grad_f_x - (X * grad_f_x).sum(dim=-1).unsqueeze(dim=-1))
        probs = torch.softmax(d_tilde.reshape(d_tilde.shape[0], -1) / 2,
                              dim=-1)
        dist = OneHotCategorical(logits=d_tilde.reshape(d_tilde.shape[0], -1) /
                                 2)
        return f_x, dist


class PottsGWGCategoricalSampler(CategoricalMetropolistHastingsSampler):
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
        super().__init__()
        self.bs = bs
        self.L = L
        self.A = A
        self.device = device

    def compute_neg_energy_and_proposal(self, X, model):
        bs, L, A, device = self.bs, self.L, self.A, self.device
        X.requires_grad_()
        f_x, grad_f_x = model.neg_energy_and_grad_f_x(X)
        with torch.no_grad():
            d_tilde = (grad_f_x - (X * grad_f_x).sum(dim=-1).unsqueeze(dim=-1))
        probs = torch.softmax(d_tilde.reshape(d_tilde.shape[0], -1) / 2,
                              dim=-1)
        dist = OneHotCategorical(logits=d_tilde.reshape(d_tilde.shape[0], -1) /
                                 2)
        return f_x, dist
