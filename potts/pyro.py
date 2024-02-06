import pyro
import torch
from pyro.nn import PyroModule, PyroParam, PyroSample
import pyro.distributions as dist
import torch.nn.functional as F
from pyro.distributions import constraints
from pyro.infer.reparam import AutoReparam

def to_wt_gauge(h, J, wt_ind):
    assert(J.ndim == 4)
    assert(h.ndim == 2)
    seq_len = len(wt_ind)
    select_positions = torch.arange(seq_len)
    J_ij_ab = J
    J_ij_ci_b = J[select_positions, :, wt_ind].unsqueeze(2)
    J_ij_a_cj = J.transpose(-1,-2)[:, select_positions, wt_ind].unsqueeze(3)
    J_ij_ci_cj = J[select_positions, :, wt_ind][:, select_positions, wt_ind].unsqueeze(2).unsqueeze(3)
    J_new = J_ij_ab - J_ij_ci_b - J_ij_a_cj + J_ij_ci_cj



    h_i_c = h[select_positions, wt_ind].unsqueeze(1)
    J_j_nequal_i = (J.transpose(-1,-2)[:, select_positions, wt_ind].unsqueeze(3) - J[select_positions, :, wt_ind][:, select_positions, wt_ind].unsqueeze(2).unsqueeze(3)).sum(dim=1).squeeze()
    J_j_equal_i = (J.transpose(-1,-2)[:, select_positions, wt_ind].unsqueeze(3) - J[select_positions, :, wt_ind][:, select_positions, wt_ind].unsqueeze(2).unsqueeze(3))[select_positions, select_positions].squeeze()
    h_new = (h - h_i_c + (J_j_nequal_i - J_j_equal_i)).to(torch.float)
    return h_new, J_new



class PottsPyro(PyroModule):
    def __init__(self, L, A, center, wt_ind=None, h_prior=None, W_prior=None, h_scale=10.0, W_scale=10.0, param_wt_gauge=False, device='cpu', dist_mask=None):
        super().__init__()
        n_params = (L*A) + int(((L*A * L*A) - (L*A))/2)
        self.device = device
        if self.device == 'cuda':
            self.cuda()

        self.L = L
        self.A = A
        self.wt_ind = wt_ind.to(self.device) if not wt_ind is None else None
        r,c = torch.triu_indices(L*A,L*A,1, device=self.device)
        self.rows = r
        self.columns = c        
        self.h_scale = h_scale
        self.W_scale = W_scale
        self.center = center.detach().clone().to(self.device)
        self.param_wt_gauge = param_wt_gauge
        dist_mask = dist_mask.to(self.device) if not dist_mask is None else None
        self.dist_mask = dist_mask
        if not self.dist_mask is None:
            self.dist_mask = self.dist_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,21,21).transpose(1,2).reshape(L*A, L*A)
            self.dist_mask = self.dist_mask[self.rows, self.columns]
        
        if h_prior is None:
            assert(W_prior is None)
            self.h_prior = torch.zeros((L, A), device=self.device)
            self.W_prior = torch.zeros((L*A, L*A), device=self.device)
        else:
            assert(not W_prior is None)
            assert(W_prior.ndim==2)
            if self.param_wt_gauge:
                W_prior = W_prior.reshape(L,A,L,A).transpose(1,2)
                h_prior, W_prior = to_wt_gauge(h_prior, W_prior, wt_ind)
                W_prior = W_prior.transpose(1,2).reshape(L*A,L*A)
            self.h_prior = h_prior.to(self.device)
            self.W_prior = W_prior.to(self.device)
            self.W_prior_triu = self.W_prior[self.rows, self.columns]

        if param_wt_gauge:
            self.h_mask = (1 - F.one_hot(self.wt_ind, num_classes=A))
            W_mask = torch.ones(L,L,A,A, device=self.device)
            for i in range(L-1):
                wt_i = self.wt_ind[i]
                for j in range(i+1, L):
                    wt_j = self.wt_ind[j]
                    W_mask[i,j,wt_i,:] = 0.0
                    W_mask[i,j,:,wt_j] = 0.0
                    W_mask[j,i,wt_j,:] = 0.0
                    W_mask[j,i,:,wt_i] = 0.0
            self.W_mask = W_mask.transpose(1,2).reshape(L*A, L*A)
        else:
            self.h_mask = 1.0
            self.W_mask = 1.0

    def params_to_W(self, W_params):
        L, A = self.L, self.A
        W = torch.zeros(L*A, L*A, device=self.device)
        W[self.rows, self.columns] = W_params
        W[self.columns, self.rows] = W_params
        return W

    #@AutoReparam()
    def model(self, X, y):
        L, A = self.L, self.A
        scale = (self.h_scale * self.h_mask) + ((1 - self.h_mask) * 1e-6)
        h_params = pyro.sample('h', dist.Normal(self.h_prior, 
                                                scale=scale).to_event(2))
                                                #scale=self.h_scale*torch.ones(L,A, device=self.device)).to_event(2))
        h = h_params * self.h_mask
        center = pyro.param('center', self.center)
        
        num_W = int(((L*A * L*A) - (L*A))/2)
        #W_params = pyro.sample('W', dist.Normal(torch.zeros(num_W), torch.ones(num_W)*self.W_scale).to_event(1))
        if self.dist_mask is None:
            W_params = pyro.sample('W', dist.Normal(self.W_prior_triu, torch.ones(num_W, device=self.device)*self.W_scale).to_event(1))
        else:
            scale = torch.ones(num_W, device=self.device)
            scale[self.dist_mask] = scale[self.dist_mask]*self.W_scale
            scale[~self.dist_mask] = scale[~self.dist_mask]*1e-6
            if self.param_wt_gauge:
                mask = self.W_mask[self.rows, self.columns] == 0.0
                scale[mask] = 1e-6
            W_params = pyro.sample('W', dist.Normal(self.W_prior_triu, scale).to_event(1))
        W = self.params_to_W(W_params)
        W = W * self.W_mask


        with pyro.plate("data"):
            if (not self.param_wt_gauge) and (not self.wt_ind is None):
                W = W.reshape(L,A,L,A).transpose(1,2)
                h, W = to_wt_gauge(h, W, self.wt_ind)
                W = W.transpose(1,2).reshape(L*A,L*A)
            energy = self.energy_from_h_W(X, h, W)
            logits = (-energy) + self.center
            pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


    def model_debug(self, X, y):
        L, A = self.L, self.A
        h_params = pyro.sample('h', dist.Normal(self.h_prior, 
                                         scale=self.h_scale*torch.ones(L,A)).to_event(2))
        h = h_params * self.h_mask
        center = pyro.param('center', self.center)
        
        num_W = int(((L*A * L*A) - (L*A))/2)
        #W_params = pyro.sample('W', dist.Normal(torch.zeros(num_W), torch.ones(num_W)*self.W_scale).to_event(1))
        W_params = pyro.sample('W', dist.Normal(self.W_prior_triu, torch.ones(num_W)*self.W_scale).to_event(1))
        W = self.params_to_W(W_params)
        W = W * self.W_mask

    def guide_map(self, X, y):
        L, A = self.L, self.A
        
        #self.center
        center_map = pyro.param("center_map", self.center)
        #center = pyro.sample('center', dist.Delta(center_map))

        h_map = pyro.param("h_mean", self.h_prior.clone())
        h = pyro.sample('h', dist.Delta(h_map).to_event(2))

        num_W = int(((L*A * L*A) - (L*A))/2)
        W_map = pyro.param("W_mean", self.W_prior_triu.clone())
        W_params = pyro.sample('W', dist.Delta(W_map).to_event(1))


    #@AutoReparam()
    def guide_mean_field(self, X, y):
        L, A = self.L, self.A
        
        #self.center
        #center_map = pyro.param("center_map", self.center)
        #center = pyro.sample('center', dist.Delta(center_map))

        h_mean = pyro.param("h_mean", self.h_prior.clone())
        h_scale = pyro.param("h_scale", (1e-2)*torch.ones(L,A, device=self.device), constraint=constraints.positive)
        h = pyro.sample('h', dist.Normal(h_mean, scale=h_scale).to_event(2))
        #h = pyro.sample('h', dist.Normal(h_mean, scale=1e-2).to_event(2))

        num_W = int(((L*A * L*A) - (L*A))/2)
        W_mean = pyro.param("W_mean", self.W_prior_triu.clone())
        W_scale = pyro.param("W_scale", torch.ones(num_W, device=self.device)*(1e-2), constraint=constraints.positive)
        W_params = pyro.sample('W', dist.Normal(W_mean, scale=W_scale).to_event(1))
        #W_params = pyro.sample('W', dist.Normal(W_mean, scale=1e-2).to_event(1))

    def guide_mean_field_debug(self, X, y):
        L, A = self.L, self.A
        
        #self.center
        #center_map = pyro.param("center_map", self.center)
        #center = pyro.sample('center', dist.Delta(center_map))

        h_mean = pyro.param("h_mean", self.h_prior.clone())
        h_scale = pyro.param("h_scale", (1e-1)*torch.ones(L,A), constraint=constraints.positive)
        h = pyro.sample('h', dist.Normal(h_mean, scale=h_scale).to_event(2))
        #h = pyro.sample('h', dist.Normal(h_mean, scale=1e-2).to_event(2))

        num_W = int(((L*A * L*A) - (L*A))/2)
        W_mean = pyro.param("W_mean", self.W_prior_triu.clone())
        W_scale = pyro.param("W_scale", torch.ones(num_W)*(1e-1), constraint=constraints.positive)
        W_params = pyro.sample('W', dist.Normal(W_mean, scale=W_scale).to_event(1))
        #W_params = pyro.sample('W', dist.Normal(W_mean, scale=1e-2).to_event(1))


        
    def to_wt_gauge(self, h, W):
        W = self.reshape_to_L_L_A_A(W)
        h, W = to_wt_gauge(h, W, self.wt_ind)
        return h, W.transpose(1,2).reshape(L*A, L*A)

    def reshape_to_L_L_A_A(self, W):
        return W.reshape(self.L, self.A, self.L, self.A).transpose(1,2)

    def energy_from_h_W(self, X, h, W):
        pairwise_energies = (torch.einsum("ij,bj->bi", W, 
                                          X.reshape(-1, self.L*self.A)) * X.reshape(-1, self.L*self.A)).sum(axis=1) / 2
        single_energies = torch.einsum('ij,bij->b', h, X)
        energies = single_energies + pairwise_energies
        return energies










class PottsPyroFromPrev(PyroModule):
    def __init__(self, L, A, center, h_mean, W_mean, h_var, W_var, device='cpu'):
        super().__init__()
        n_params = (L*A) + int(((L*A * L*A) - (L*A))/2)
        self.device = device
        if self.device == 'cuda':
            self.cuda()

        self.L = L
        self.A = A
        r,c = torch.triu_indices(L*A,L*A,1, device=self.device)
        self.rows = r
        self.columns = c        
        self.center = center.detach().clone().to(self.device)
        self.h_mean = h_mean.clone().to(device)
        self.h_scale = h_var.clone().to(device)
        self.W_mean = W_mean.to(device).clone()

        self.W_mean_triu = W_mean.to(device)[self.rows, self.columns].clone()
        self.W_scale_triu = W_var.to(device)[self.rows, self.columns].clone()


    #@AutoReparam()
    def model(self, X, y):
        L, A = self.L, self.A
        h = pyro.sample('h', dist.Normal(self.h_mean, 
                                         scale=self.h_scale).to_event(2))
        center = self.center
        num_W = int(((L*A * L*A) - (L*A))/2)
        W_params = pyro.sample('W', dist.Normal(self.W_mean_triu, self.W_scale_triu).to_event(1))
        W = self.params_to_W(W_params)
        with pyro.plate("data"):
            energy = self.energy_from_h_W(X, h, W)
            logits = (-energy) + self.center
            pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    def params_to_W(self, W_params):
        L, A = self.L, self.A
        W = torch.zeros(L*A, L*A, device=self.device)
        W[self.rows, self.columns] = W_params
        W[self.columns, self.rows] = W_params
        return W

    def guide_map(self, X, y):
        L, A = self.L, self.A
        
        h_map = pyro.param("h_mean", self.h_mean.clone())
        h = pyro.sample('h', dist.Delta(h_map).to_event(2))

        num_W = int(((L*A * L*A) - (L*A))/2)
        W_map = pyro.param("W_mean", self.W_mean_triu.clone())
        W_params = pyro.sample('W', dist.Delta(W_map).to_event(1))

    def guide_mean_field(self, X, y):
        L, A = self.L, self.A
        
        h_mean = pyro.param("h_mean", self.h_mean.clone())
        h_scale = pyro.param("h_scale", self.h_scale.clone(), constraint=constraints.positive)
        h = pyro.sample('h', dist.Normal(h_mean, scale=h_scale).to_event(2))

        num_W = int(((L*A * L*A) - (L*A))/2)
        W_mean = pyro.param("W_mean", self.W_mean_triu.clone())
        W_scale = pyro.param("W_scale", self.W_scale_triu.clone(), constraint=constraints.positive)
        W_params = pyro.sample('W', dist.Normal(W_mean, scale=W_scale).to_event(1))
        #W_params = pyro.sample('W', dist.Normal(W_mean, scale=1e-2).to_event(1))
        
    def to_wt_gauge(self, h, W):
        W = self.reshape_to_L_L_A_A(W)
        h, W = to_wt_gauge(h, W, self.wt_ind)
        return h, W.transpose(1,2).reshape(L*A, L*A)

    def reshape_to_L_L_A_A(self, W):
        return W.reshape(self.L, self.A, self.L, self.A).transpose(1,2)

    def energy_from_h_W(self, X, h, W):
        pairwise_energies = (torch.einsum("ij,bj->bi", W, 
                                          X.reshape(-1, self.L*self.A)) * X.reshape(-1, self.L*self.A)).sum(axis=1) / 2
        single_energies = torch.einsum('ij,bij->b', h, X)
        energies = single_energies + pairwise_energies
        return energies

