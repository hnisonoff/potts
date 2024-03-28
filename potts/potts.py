import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
import warnings
from torch import Tensor, autograd





class Symmetric(nn.Module):

    def __init__(self, L, A):
        super().__init__()
        # mask out diagonal
        mask = (torch.ones(L, L) - torch.eye(L))[:, :, None, None]
        mask = (torch.ones(L, L, A, A) * mask).transpose(1, 2).reshape(
            L * A, L * A)
        self.register_buffer('mask', mask)

    def forward(self, X):
        return (X.triu() + X.triu(1).transpose(-1, -2)) * self.mask


class WeightMask(nn.Module):

    def __init__(self, W_mask):
        super().__init__()
        self.register_buffer('W_mask', W_mask)

    def forward(self, X):
        return X * self.W_mask


class WTGauge(nn.Module):

    def __init__(self, W_mask):
        super().__init__()
        self.register_buffer('W_mask', W_mask)

    def forward(self, X):
        return X * self.W_mask


class Potts(torch.nn.Module):

    def __init__(self,
                 L=None,
                 A=None,
                 h=None,
                 W=None,
                 temp=1.0,
                 wt_enc=None,
                 weight_mask=None):
        super().__init__()
        if L is not None:
            assert (A is not None) and (h is None) and (W is None)
        if A is not None:
            assert (L is not None) and (h is None) and (W is None)
        if h is not None:
            assert (W is not None) and (L is None) and (A is None)
        if W is not None:
            assert (W is not None) and (L is None) and (A is None)
        # Scenario 1
        if (L is not None) and (A is not None):
            self.L = L
            self.A = A
            self.h = nn.Parameter(torch.zeros(L * A))
            self.W = nn.Linear(L * A, L * A, bias=False)
            self.W.weight.data.fill_(0.0)
            self.sym = Symmetric(L, A)
            parametrize.register_parametrization(self.W, "weight", self.sym)
        else:
            h = h.clone().detach().to(torch.float)
            #h = torch.tensor(h, dtype=torch.float)
            assert (h.ndim == 2)
            self.L, self.A = h.shape
            self.h = nn.Parameter(h.reshape(-1))
            W = W.clone().detach().to(torch.float)
            #W = torch.tensor(W, dtype=torch.float)
            assert (W.ndim == 2)
            assert W.shape[0] == W.shape[1] == self.L * self.A
            self.W = nn.Linear(self.L * self.A, self.L * self.A, bias=False)
            self.W.weight = nn.Parameter(W)
            if False:
                self.sym = Symmetric(self.L, self.A)
                parametrize.register_parametrization(self.W, "weight",
                                                     self.sym)
            #self.W.weight = nn.Parameter(W)

        self.temp = temp
        self.wt_enc = wt_enc
        if not (wt_enc is None):
            h_mask = torch.ones((self.L, self.A))
            for pos in range(self.L):
                wt_aa_idx = wt_enc[pos]
                h_mask[pos, wt_aa_idx] = 0
            h_mask = h_mask.reshape(-1)
            self.register_buffer('h_mask', h_mask)

            W_mask = torch.ones((self.L, self.L, self.A, self.A))
            for pos_i in range(self.L):
                for pos_j in range(self.L):
                    if pos_i == pos_j:
                        W_mask[pos_i, pos_j] = 0.0
                    else:
                        wt_aa_i = wt_enc[pos_i]
                        wt_aa_j = wt_enc[pos_j]
                        W_mask[pos_i, pos_j, wt_aa_i] = 0.0
                        W_mask[pos_i, pos_j, :, wt_aa_j] = 0.0
                        W_mask[pos_j, pos_i, wt_aa_j] = 0.0
                        W_mask[pos_j, pos_i, :, wt_aa_i] = 0.0
            W_mask = W_mask.transpose(1, 2).reshape(self.L * self.A,
                                                    self.L * self.A)

            self.wt_gauge = WTGauge(W_mask)
            parametrize.register_parametrization(self.W, "weight",
                                                 self.wt_gauge)

        if not (weight_mask is None):
            self.weight_mask = WeightMask(weight_mask)
            parametrize.register_parametrization(self.W, "weight",
                                                 self.weight_mask)

    def load_from_weights(self, h, W):
        h = h.clone().detach().to(torch.float)
        assert (h.ndim == 2)
        self.L, self.A = h.shape
        self.h = nn.Parameter(h.reshape(-1))
        W = W.clone().detach().to(torch.float)
        assert (W.ndim == 2)
        assert W.shape[0] == W.shape[1] == self.L * self.A
        self.W = nn.Linear(self.L * self.A, self.L * self.A, bias=False)
        self.W.weight = nn.Parameter(W)
        self.sym = Symmetric(self.L, self.A)
        parametrize.register_parametrization(self.W, "weight", self.sym)
        return

    def reshape_to_L_L_A_A(self):
        return self.W.weight.reshape(
            (self.L, self.A, self.L, self.A)).transpose(1, 2)

    def pseudolikelihood(self, X, mask=None):
        if not (self.wt_enc is None):
            tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) +
                     (self.h * self.h_mask))).reshape(-1, self.L, self.A)
        else:
            tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) + self.h)).reshape(
                -1, self.L, self.A)
        tmp = X * F.log_softmax(tmp, dim=2)
        if not (mask is None):
            tmp = tmp * mask.unsqueeze(-1)
        return tmp.reshape(-1, self.L * self.A).sum(dim=1)

    def marginals(self, X, mask=None):
        tmp = (-((self.W(torch.flatten(X, 1, 2)) / 2) + self.h)).reshape(
            -1, self.L, self.A)
        tmp = F.log_softmax(tmp, dim=2)
        return tmp

    def pseudolikelihood_debug(self, X, mask=None):
        tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) + self.h)).reshape(
            -1, self.L, self.A)
        return F.log_softmax(tmp, dim=2)

    def forward(self, X, beta=1.0):
        energy = X * ((self.W(torch.flatten(X, 1, 2)) * beta / 2.0) +
                      self.h).reshape(-1, self.L, self.A)
        return energy.reshape(-1, self.L * self.A).sum(dim=1)

    def single_pairwise_energy(self, X, beta=1.0):
        pairwise = (X * self.W(torch.flatten(X, 1, 2)).reshape(
            -1, self.L, self.A)).reshape(
                -1, self.L * self.A).sum(dim=1) * beta / 2.0
        single = (X * self.h.reshape(-1, self.L, self.A)).reshape(
            -1, self.L * self.A).sum(dim=1)
        return single, pairwise

    def grad_f_x(self, X):
        '''
        returns shape batch x length x num aa
        '''
        return (-((self.W(torch.flatten(X, 1, 2)) + self.h))).reshape(
            -1, self.L, self.A) / self.temp

    def neg_energy_and_grad_f_x(self, X, beta=1.0):
        '''
        returns shape batch x length x num aa
        '''
        tmp = self.W(torch.flatten(X, 1, 2))
        grad_f_x = -((tmp * beta + self.h))
        neg_energy = -X * (
            (tmp * beta / 2.0) + self.h).reshape(-1, self.L, self.A)
        neg_energy = neg_energy.reshape(-1, self.L * self.A).sum(dim=1)
        grad_f_x = grad_f_x.reshape(-1, self.L, self.A)
        return neg_energy / self.temp, grad_f_x / self.temp

    def energy_wt_gauge(self, X, wt_enc):
        L, A = X.shape[1:]
        h = self.h
        J = self.W.weight
        h_wt, J_wt = to_wt_gauge(h.reshape(L, A), 
                                 J.reshape(L,A,L,A).transpose(1,2), 
                                 wt_enc)
        pairwise_energies = (torch.einsum("ij,bj->bi", J_wt.transpose(1,2).reshape(L*A, L*A), 
                                          X.reshape(-1, L*A)) * X.reshape(-1, L*A)).sum(axis=1) / 2
        single_energies = torch.einsum('ij,bij->b', h_wt, X)
        energies = single_energies + pairwise_energies
        return energies


# def to_wt_gauge(h, J, wt_ind):
#     '''
#     Thank you Akosua!!!!
#     '''
#     assert (J.ndim == 4)
#     assert (h.ndim == 2)
#     select_positions = torch.arange(len(wt_ind))
#     J_ij_ab = J
#     J_ij_ci_b = J[select_positions, :, wt_ind].unsqueeze(2)
#     J_ij_a_cj = J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3)
#     J_ij_ci_cj = J[select_positions, :,
#                    wt_ind][:, select_positions,
#                            wt_ind].unsqueeze(2).unsqueeze(3)
#     J_new = J_ij_ab - J_ij_ci_b - J_ij_a_cj + J_ij_ci_cj

#     h_i_c = h[select_positions, wt_ind].unsqueeze(1)
#     J_j_nequal_i = (
#         J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
#         J[select_positions, :, wt_ind][:, select_positions,
#                                        wt_ind].unsqueeze(2).unsqueeze(3)).sum(
#                                            dim=1).squeeze()
#     J_j_equal_i = (
#         J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
#         J[select_positions, :, wt_ind][:, select_positions,
#                                        wt_ind].unsqueeze(2).unsqueeze(3)
#     )[select_positions, select_positions].squeeze()
#     h_new = (h - h_i_c + (J_j_nequal_i - J_j_equal_i)).to(torch.float)
#     return h_new, J_new

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

def to_zs_gauge(h, J):
    assert (J.ndim == 4)
    assert (h.ndim == 2)
    L, A = h.shape
    J_zs = (J - J.mean(dim=2).unsqueeze(dim=2) -
            J.mean(dim=3).unsqueeze(dim=3) +
            (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3)))
    tmp = (
        J.mean(dim=3).unsqueeze(dim=3) -
        (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3))).sum(
            dim=1).squeeze()
    h_zs = (h - h.mean(dim=-1).unsqueeze(1) + tmp)
    return h_zs, J_zs



class L_BFGS(object):
    def __init__(self, x, g, history_size=100):
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = 1.
        self.alpha = x.new_empty(history_size)
        self.history_size = history_size
        self.x_prev = x.clone(memory_format=torch.contiguous_format)
        self.g_prev = g.clone(memory_format=torch.contiguous_format)
        self.n_updates = 0

    def solve(self, d):
        mem_size = len(self.y)
        dshape = d.shape
        d = d.view(-1).clone(memory_format=torch.contiguous_format)
        for i in reversed(range(mem_size)):
            self.alpha[i] = self.s[i].dot(d) * self.rho[i]
            d.add_(self.y[i], alpha=-self.alpha[i])
        d.mul_(self.H_diag)
        for i in range(mem_size):
            beta_i = self.y[i].dot(d) * self.rho[i]
            d.add_(self.s[i], alpha=self.alpha[i] - beta_i)

        return d.view(dshape)

    def update(self, x, g):
        s = (x - self.x_prev).view(-1)
        y = (g - self.g_prev).view(-1)
        rho_inv = y.dot(s)
        if rho_inv <= 1e-10:
            # curvature is negative; do not update
            return
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)
        self.x_prev.copy_(x, non_blocking=True)
        self.g_prev.copy_(g, non_blocking=True)
        self.n_updates += 1


def project(x, y):
    return x.masked_fill(x.sign() != y.sign(), 0)


def pseudo_grad(x, grad_f, alpha):
    grad_r = alpha * x.sign()
    grad_right = grad_f + grad_r.masked_fill(x == 0, alpha)
    grad_left = grad_f + grad_r.masked_fill(x == 0, -alpha)
    pgrad = torch.zeros_like(x)
    pgrad = torch.where(grad_right < 0, grad_right, pgrad)
    pgrad = torch.where(grad_left > 0, grad_left, pgrad)
    return pgrad


def backtracking(dir_evaluate, x, v, t, d, f, tol=0.1, decay=0.95, maxiter=500):
    for n_iter in range(1, maxiter + 1):
        x_new, f_new = dir_evaluate(x, t, d)
        if f_new <= f - tol * v.mul(x_new-x).sum():
            break
        t = t * decay
    else:
        warnings.warn('line search did not converge.')

    return t, n_iter


@torch.no_grad()
def owlqn(fun, x0, alpha=1., lr=1, max_iter=20, xtol=1e-5, history_size=100,
          line_search='brent', ls_options=None, verbose=0):
    """Orthant-wise limited-memory quasi-newton

    Parameters
    ----------
    fun : callable
        Objective function. Must output a scalar with grad_fn
    x0 : Tensor
        Initial value of the parameters
    alpha : float
        Sparsity weight of the Lasso problem
    lr : float
        Learning rate (default = 1)
    max_iter : int
        Maximum number of iterations (default = 20)
    xtol : float
        Termination tolerance on parameter changes
    history_size : int
        History size for L-BFGS memory updates (default = 100)
    line_search : str, optional
        Optional line search specifier

    Returns
    -------
    x : Tensor
        Final value of the parameters after optimization.

    """
    #assert x0.dim() == 2
    from scipy.optimize import minimize_scalar
    verbose = int(verbose)
    if ls_options is None:
        ls_options = {}

    def evaluate(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = fun(x)
        # NOTE: do not include l1 penalty term in the gradient
        grad = autograd.grad(f, x)[0]
        f = f.detach() + alpha * x.norm(p=1)
        grad_pseudo = pseudo_grad(x, grad, alpha)
        return f, grad, grad_pseudo

    # evaluate initial f(x) and f'(x)
    x = x0.detach()
    f, g, g_pseudo = evaluate(x)
    if verbose:
        print('initial f: %0.4f' % f)

    # initialize
    lbfgs = L_BFGS(x, g, history_size)
    t = torch.clamp(lr / g_pseudo.norm(p=1), max=lr)
    delta_x = x.new_tensor(float('inf'))

    # optimize for a max of max_iter iterations
    for n_iter in range(1, max_iter + 1):

        # descent direction
        v = g_pseudo.neg()

        # compute quasi-newton direction
        d = lbfgs.solve(v)

        # project the quasi-newton direction
        d = project(d, v)

        # compute eta
        eta = torch.where(x == 0, v.sign(), x.sign())

        # perform line search to determine step size
        if line_search == 'brent':
            def line_obj(t):
                x_new = project(x.add(d, alpha=t), eta)
                f_new = fun(x_new) + alpha * x_new.norm(p=1)
                return float(f_new)

            res = minimize_scalar(line_obj, bounds=(0,10), method='bounded')
            t = res.x
            ls_iters = res.nfev

        elif line_search == 'backtrack':
            def dir_evaluate(x, t, d):
                x_new = project(x.add(d, alpha=t), eta)
                f_new = fun(x_new) + alpha * x_new.norm(p=1)
                return x_new, f_new

            t, ls_iters = backtracking(dir_evaluate, x, v, t, d, f, **ls_options)

        elif line_search == 'none':
            ls_iters = 0

        else:
            raise RuntimeError

        # update x
        x_new = project(x.add(d, alpha=t), eta)
        torch.norm(x_new - x, p=2, out=delta_x)
        x = x_new

        # re-evaluate
        f, g, g_pseudo = evaluate(x)
        if verbose > 1:
            print('iter %3d - ls_iters: %3d - f: %0.4f - dx: %0.3e'
                  % (n_iter, ls_iters, f, delta_x))
        # check for convergence
        if delta_x <= xtol:
            break

        # update hessian estimate
        lbfgs.update(x, g)
        t = lr

    if verbose:
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)

    return x



class PottsL1(torch.nn.Module):
    def __init__(self, L, A, wt_ind=None):
        super().__init__()
        n_params = (L*A) + int(((L*A * L*A) - (L*A))/2)
        self.L = L
        self.A = A
        self.p = nn.Parameter(torch.zeros(n_params))
        self._h = torch.zeros(L, A)
        self._W = torch.zeros((L*A, L*A))
        r,c = torch.triu_indices(L*A,L*A,1)
        self.rows = r
        self.columns = c
        self.wt_ind = wt_ind

    def _get_h_W(self):
        L, A = self.L, self.A
        self._h = self.p[:L * A].reshape(L,A)
        self._W[self.rows, self.columns] = self.p[L*A:]
        self._W[self.columns, self.rows] = self.p[L*A:]
        if self.wt_ind is None:
            return self._h, self._W
        W = self.reshape_to_L_L_A_A(self._W)
        h, W = to_wt_gauge(self._h, W, self.wt_ind)
        return h, W.transpose(1,2).reshape(L*A, L*A)


    def reshape_to_L_L_A_A(self, W):
        return W.reshape(self.L, self.A, self.L, self.A).transpose(1,2)

    def forward(self, X):
        return self.energy(X)

    def energy(self, X):
        h, W = self._get_h_W()
        pairwise_energies = (torch.einsum("ij,bj->bi", W, 
                                          X.reshape(-1, 167*21)) * X.reshape(-1, 167*21)).sum(axis=1) / 2
        single_energies = torch.einsum('ij,bij->b', h, X)
        energies = single_energies + pairwise_energies
        return energies


    def get_h_W_from_p(self, p):
        L, A = self.L, self.A
        self._h = p[:L * A].reshape(L,A)
        self._W[self.rows, self.columns] = p[L*A:]
        self._W[self.columns, self.rows] = p[L*A:]
        if self.wt_ind is None:
            return self._h, self._W
        W = self.reshape_to_L_L_A_A(self._W)
        h, W = to_wt_gauge(self._h, W, self.wt_ind)
        return h, W.transpose(1,2).reshape(L*A, L*A)



    def energy_from_p(self, X, p):
        L, A = self.L, self.A
        self._h = p[:L * A].reshape(L,A)
        self._W[self.rows, self.columns] = p[L*A:]
        self._W[self.columns, self.rows] = p[L*A:]
        if self.wt_ind is None:
            return self._h, self._W
        W = self.reshape_to_L_L_A_A(self._W)
        h, W = to_wt_gauge(self._h, W, self.wt_ind)
        W = W.transpose(1,2).reshape(L*A, L*A)
        pairwise_energies = (torch.einsum("ij,bj->bi", W, 
                                          X.reshape(-1, 167*21)) * X.reshape(-1, 167*21)).sum(axis=1) / 2
        single_energies = torch.einsum('ij,bij->b', h, X)
        energies = single_energies + pairwise_energies
        return energies

    
def get_subset_potts(h, W, idxs_to_keep, wt_ind):
    assert(h.ndim == 2)
    assert(W.ndim == 2)
    L, A = h.shape
    W = W.reshape(L,A,L,A).transpose(1,2)

    sub_L = len(idxs_to_keep)
    sub_h = h[idxs_to_keep].clone()
    h_tilde = torch.zeros_like(sub_h)
    other_idxs = [i for i in range(L) if i not in idxs_to_keep]
    other_aas = [wt_ind[i] for i in other_idxs]

    # the constant energy from everything we are keeping fixed
    # includes single site energies and constant pairwise energies
    other_ss_energy = h[other_idxs, other_aas].sum()
    for i in other_idxs:
        aa_i = wt_ind[i]
        for j in other_idxs:
            if j <= i:
                continue
            aa_j = wt_ind[j]
            other_ss_energy += W[i,j,aa_i,aa_j]

    # add the constant energy evenly to all single-site terms
    h_tilde += other_ss_energy / len(idxs_to_keep)

    new_pos = 0
    for i in idxs_to_keep:
        for aa_i in range(A):
            for j, aa_j in zip(other_idxs, other_aas):
                h_tilde[new_pos, aa_i] += W[i, j, aa_i, aa_j]
        new_pos += 1

    new_h = sub_h + h_tilde
    sub_W = W[idxs_to_keep][:, idxs_to_keep]
    sub_W = sub_W.transpose(1,2).reshape(sub_L * A, sub_L * A)
    return new_h, sub_W


def get_subset_potts(model, idxs_to_keep, wt_ind):
    L, A = model.L, model.A
    h = model.h.reshape(L, A)
    W = model.W.weight
    h_new, W_new = get_subset_potts_optimized(h, W, idxs_to_keep, wt_ind)
    sub_potts = Potts(h=h_new, W=W_new)
    return sub_potts

def get_subset_potts_optimized(h, W, idxs_to_keep, wt_ind):
    assert(h.ndim == 2)
    assert(W.ndim == 2)
    
    L, A = h.shape
    W = W.reshape(L, A, L, A).transpose(1, 2)

    sub_L = len(idxs_to_keep)
    sub_h = h[idxs_to_keep]

    h_tilde = torch.zeros_like(sub_h)

    other_idxs = [i for i in range(L) if i not in idxs_to_keep]
    other_aas = [wt_ind[i] for i in other_idxs]

    # Precompute constants
    num_idxs_to_keep = len(idxs_to_keep)
    other_ss_energy = h[other_idxs, other_aas].sum()

    # Replace the nested loops with tensor operations
    i_tensor = torch.tensor(other_idxs)
    j_tensor = torch.tensor(other_idxs)

    aa_i_tensor = torch.tensor([wt_ind[i] for i in other_idxs])
    aa_j_tensor = torch.tensor([wt_ind[j] for j in other_idxs])

    # Use torch.triu_indices to get the upper triangular indices
    upper_triangular_indices = torch.triu_indices(len(other_idxs), len(other_idxs), offset=1)

    i_upper_triangular = i_tensor[upper_triangular_indices[0]]
    j_upper_triangular = j_tensor[upper_triangular_indices[1]]
    aa_i_upper_triangular = aa_i_tensor[upper_triangular_indices[0]]
    aa_j_upper_triangular = aa_j_tensor[upper_triangular_indices[1]]

    # Use tensor indexing to calculate other_ss_energy
    other_ss_energy += torch.sum(W[i_upper_triangular, j_upper_triangular, aa_i_upper_triangular, aa_j_upper_triangular])

    # Add the constant energy evenly to all single-site terms
    h_tilde += other_ss_energy / num_idxs_to_keep

    new_pos = 0
    for i in idxs_to_keep:
        for aa_i in range(A):
            h_tilde[new_pos, aa_i] += torch.sum(W[i, other_idxs, aa_i, other_aas])
        new_pos += 1

    new_h = sub_h + h_tilde
    sub_W = W[idxs_to_keep][:, idxs_to_keep]
    sub_W = sub_W.transpose(1, 2).reshape(sub_L * A, sub_L * A)

    return new_h, sub_W
