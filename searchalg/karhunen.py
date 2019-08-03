import numpy as np
import GPy
import scipy

__all__ = ['KarhunenLoeveExpansion']

class KarhunenLoeveExpansion(object):
    def __init__(self, kernel, Xq = None, nq = 100, alpha = 0.95):
        self.kernel = kernel
        if Xq == None:
            Xq = np.linspace(0,1,nq)[:,None]
        self.Xq = Xq
        self.nq = Xq.shape[0]
        Kq = kernel.K(Xq)
        lam, u = scipy.linalg.eigh(Kq, overwrite_a=True)
        lam = lam[::-1]
        lam[lam <= 0.] = 0.
        energy = np.cumsum(lam) / np.sum(lam)
        i_end = np.arange(energy.shape[0])[energy > alpha][0] + 1
        u = u[:, ::-1]
        self.lam = lam[:i_end]
        self.u = u[:, :i_end]
        self.sqrt_lam = np.sqrt(self.lam)
        self.energy = energy
        self.num_xi = i_end
    def eval_phi(self, xtest):
        Kc = self.kernel.K(xtest, self.Xq)
        self.phi = np.einsum('i, ji, rj ->ri', 1.0/self.lam, self.u, Kc)
        return self.phi
    def __call__(self, xtest, xi):
        return np.dot(self.phi, xi * self.sqrt_lam)
