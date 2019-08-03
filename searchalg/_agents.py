''' This code evaluates the conergence of the 
    Bayesian Global Optimization
'''
__all__ = ['SkillfulAgent', 'NaiveAgent']

from mpi4py import MPI
import pickle
import numpy as np
import GPy
import GPyOpt
import sys
from karhunen import *
from collections import namedtuple
from itertools import product
import itertools


class EmbarrassParallel(object):
    '''
    Define the methods to run the search in parallel
    num_samples: Number of random function
    size: the number of the cpu's
    '''

    def __init__(self, n_samples, size):
        
        self.n_samples = n_samples
        self.size      = size

    def split(self):

        jobs = list(range(self.n_samples))
        return [jobs[i::self.size] for i in range(self.size)]




class SkillfulAgent(EmbarrassParallel):

    '''
    Evaluates how a skillful agent improves the quality of the search

    lengthscale: Smoothness of the drawn functions
    max_time: maximum time to evaluate the BGO
    max_iter: maximum number of iterations in BGO
    d: the dimension of the functiuons
    n: number of test points
    '''

    def __init__(self, lengthscale, variance, num_initial_design, max_time, max_iter, d, n, energy, eps, n_samples, size, filename):

        self.max_time           = max_time
        self.max_iter           = max_iter
        self.d                  = d
        self.n                  = n
        self.lengthscale        = lengthscale
        self.variance           = variance
        self.num_initial_design = num_initial_design
        self.energy             = energy
        self.eps                = eps
        self.filename           = filename
        super().__init__(n_samples, size)


    def bayes_opt(self, karhunen, indc, domain = (0, 1), 
                  type_initial_design = 'random', acquisition = 'EI', 
                  normalize_Y = True, exact_feval = True):

        bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': domain, 
                  'dimensionality':self.d}]
        xi   = np.random.randn(len(indc))
        myBoptnD = GPyOpt.methods.BayesianOptimization(f=lambda x, ksi=xi, kle = karhunen, ic=indc : 
                                                         -self.f_ndsample(kle, ksi, x, ic),
                                                       domain=bounds,
                                                       initial_design_numdata = self.num_initial_design,
                                                       initial_design_type=type_initial_design,
                                                       acquisition_type=acquisition,
                                                       normalize_Y = normalize_Y,
                                                       exact_feval = exact_feval)
        myBoptnD.run_optimization(self.max_iter, self.max_time, eps=self.eps)
        return -myBoptnD.Y_best

    def f_ndsample(self, kle, xi, x, indc):

        sum = 0.0
        phi = np.ndarray(shape=(x.shape[0],len(indc)))
        PHI = []
        for d in range(len(indc[0])):
            PHI.append(kle.eval_phi(x[:,d].flatten()[:,None]))
        count = 0
        PHI = np.array(PHI[0])
        dim = len(indc[0])
        for tt in range(len(indc)):
            ic = indc[tt]
            temp = 1.0
            for d in range(dim):
                temp *= PHI[:,ic[d]]*kle.sqrt_lam[ic[d]]
            phi[:,count] = temp
            count += 1
        return np.dot(phi, xi)

    def run_parallel(self):

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        k = GPy.kern.RBF(1,variance = self.variance, lengthscale=self.lengthscale)
        if rank == 0:
            kle = KarhunenLoeveExpansion(k, nq=self.n, alpha = self.energy)
            indc = list(product(range(kle.num_xi), repeat=self.d))
            q_l = []
            q_m = []
            q_u = []
            jobs = self.split()
        else:
            jobs = None
            kle = None
            indc = None
        jobs = comm.scatter(jobs, root=0)
        kle = comm.bcast(kle, root=0)
        indc = comm.bcast(indc, root=0)
        results = []
        for job in jobs:
            xi = np.random.randn(len(indc))
            results.append(self.bayes_opt(kle, indc))

        results = comm.gather(results, root=0)
        if rank == 0:
            with open(self.filename, 'wb') as f:
                pickle.dump(results, f)

class NaiveAgent(SkillfulAgent):

    def __init__(self, lengthscale, variance, num_initial_design, max_time, max_iter, d, n, energy, eps, n_samples, size, filename):
        ''' 
        Evaluates how a naive agent improves the quality with random search
        '''

        self.RBDesignProblem    = namedtuple('RBDesignProblem', ['a'])
        super().__init__(lengthscale, variance, num_initial_design, max_time, max_iter, d, n, energy, eps, n_samples, size, filename)

    def quality_hrc(self, e, rbdp):

        if e == 0:
            return None
        X_e = np.random.rand(e)
        A_e = [rbdp.a(np.array([x])[:,None]) for x in X_e]
        T_e = np.argmax(A_e)
        return (X_e[T_e], A_e[T_e])

    def random_search(self, karhunen, xi, indc):

        f=lambda x, ksi=xi, kle = karhunen, ic = indc : self.f_ndsample(kle, ksi, x, ic)
        rbdp = self.RBDesignProblem(f)
        q_m = []
        q_u = []
        q_l = []
        A_es = [self.quality_hrc(1, rbdp)[1][0]]
        for e in range(1, self.max_iter):
            new_query = self.quality_hrc(1, rbdp)[1][0]
            if new_query >= A_es[-1]:
                A_es += [new_query]
            else:
                A_es += [A_es[-1]]
        return np.array(A_es)

        def f_sample(self, kle, xi, x):
            PHI = 1
            for i in range(d):
                PHI *= kle.eval_phi(x[:,i][:,None]) 
            kle.phi = PHI
            return kle(x,xi)

    def run_parallel(self):

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        k = GPy.kern.RBF(1,variance = self.variance, lengthscale=self.lengthscale)
        if rank == 0:
            kle = KarhunenLoeveExpansion(k, nq=self.n, alpha = self.energy)
            indc = list(product(range(kle.num_xi), repeat=self.d))
            q_l = []
            q_m = []
            q_u = []
            jobs = self.split()
        else:
            jobs = None
            kle = None
            indc = None
        jobs = comm.scatter(jobs, root=0)
        kle = comm.bcast(kle, root=0)
        indc = comm.bcast(indc, root=0)
        results = []
        for job in jobs:
            xi = np.random.randn(len(indc))
            results.append(self.random_search(kle, xi, indc))
        results = comm.gather(results, root=0)
        if rank == 0:
            with open(self.filename, 'wb') as f:
                pickle.dump(results, f)


if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.size
    n_samples = 400
    filename1  = "./skillful.out"
    filename2  = "./naive.out"
    Agent = SkillfulAgent(0.1, 1.0, 2, 2000, 100, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename1)
    naive_agent = NaiveAgent(0.1, 1.0, 2, 2000, 40, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename2)

