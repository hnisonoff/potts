import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F


class Symmetric(nn.Module):

    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class Potts(torch.nn.Module):

    def __init__(self, L, A):
        super(Potts, self).__init__()
        self.L = L
        self.A = A
        self.h = nn.Parameter(torch.zeros(L * A))
        self.W = nn.Linear(L * A, L * A, bias=False)
        parametrize.register_parametrization(self.W, "weight", Symmetric())

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
            -1, self.L, self.A)

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
        return neg_energy, grad_f_x
