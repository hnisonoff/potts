import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F


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


class Potts(torch.nn.Module):

    def __init__(self, L=None, A=None, h=None, W=None, temp=1.0):
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
            self.sym = Symmetric(L, A)
            parametrize.register_parametrization(self.W, "weight", self.sym)
        else:
            h = torch.tensor(h, dtype=torch.float)
            assert (h.ndim == 2)
            self.L, self.A = h.shape
            self.h = nn.Parameter(h.reshape(-1))
            W = torch.tensor(W, dtype=torch.float)
            assert (W.ndim == 2)
            assert W.shape[0] == W.shape[1] == self.L * self.A
            self.W = nn.Linear(self.L * self.A, self.L * self.A, bias=False)
            self.W.weight = nn.Parameter(W)

        self.temp = temp

    def reshape_to_L_L_A_A(self):
        return self.W.weight.reshape(
            (self.L, self.A, self.L, self.A)).transpose(1, 2)

    def pseudolikelihood(self, X):
        tmp = (-(self.W(torch.flatten(X, 1, 2)) + self.h)).reshape(
            -1, self.L, self.A)
        tmp = X * F.log_softmax(tmp, dim=2)
        return tmp.reshape(-1, self.L * self.A).sum(dim=1)

    def forward(self, X, beta=1.0):
        energy = X * ((self.W(torch.flatten(X, 1, 2)) * beta / 2.0) +
                      self.h).reshape(-1, self.L, self.A)
        return energy.reshape(-1, self.L * self.A).sum(dim=1)

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
