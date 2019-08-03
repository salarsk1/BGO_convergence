import sys
sys.path.append('../')
from searchalg import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
n_samples = 2
lengthscale_skill = [0.005, 0.01, 0.05, 0.1]

for l in lengthscale_skill:
    filename  = "/Users/salarsk/developments/my_codes/examples/"+"skillful"+str(l)+".out"
    skillful_agent = SkillfulAgent(l, 1.0, 2, 2000, 100, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename)
    skillful_agent.run_parallel()

lengthscale_naive = [0.01, 0.03, 0.05, 0.1]
for l in lengthscale_naive    
    filename  = "/Users/salarsk/developments/my_codes/examples/"+"naive"+str(l)+".out"
    naive_agent = NaiveAgent(0.1, 1.0, 2, 2000, 40, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename)
    naive_agent.run_parallel()