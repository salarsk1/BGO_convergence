from searchalg import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
n_samples = 400
filename1  = "/Users/salarsk/developments/my_codes/skillful.out"
filename2  = "/Users/salarsk/developments/my_codes/naive.out"
Agent = SkillfulAgent(0.1, 1.0, 2, 2000, 100, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename1)
naive_agent = NaiveAgent(0.1, 1.0, 2, 2000, 40, 1, 1000, 0.95, 1.0e-6, n_samples, size, filename1)