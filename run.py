import hamil
import utils
import encode_ucc_wavefunction as euw

from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator, FermionOperator, normal_ordered
from openfermionpyscf import run_pyscf
from scipy.sparse import csr_matrix
import numpy as np
import cProfile
import pstats
import time
from sympy import Symbol
import qc_qmc
import numpy as np
import cProfile
import pstats
import pickle

nqubits = 16 
ref = (0,1,2,3,4,5,6,7)
pop_thresh = 2 
dt = 0.1
#dets = [(0,1), (2,3),(4,5),(0,5),(1,4)]
#dets = [(1, 2, 3, 4), (2, 3, 4, 5), (1, 3, 4, 6), (1, 2, 5, 6), (1, 2, 4, 7), (2, 3, 6, 7), (0, 2, 3, 5), (0, 3, 5, 6), (0, 3, 4, 7), (0, 2, 5, 7), (0, 1, 3, 6), (0, 1, 4, 5), (0, 1, 6, 7), (0, 1, 2, 7), (0, 1, 2, 3)]
dets = np.genfromtxt('pqe/H8_DETS', dtype=int)
for i in range(len(dets)):
    for j in range(6):
        dets[i][j] -= 1
dets = [tuple(x) for x in dets]
for r in np.linspace(1.0,3.0,11):
    x = r
    geometry = [['H', [0,0,0]], ['H', [x,0,0]], ['H', [2*x,0,0]], ['H', [3*x, 0, 0]], ['H',[4*x, 0, 0]],['H',[5*x, 0, 0]], ['H', [6*x,0,0]], ['H', [7*x, 0, 0]]]
    basis = 'sto-3g'
    multiplicity =1 
    charge = 0
    
    hf_e, fci_e, qubit_h = hamil.get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    
    ferm_h = normal_ordered(get_fermion_operator(molecule.get_molecular_hamiltonian()))#occupied_indices = [0], active_indices = [1,2,3,4,5,6])))
    for nshots in [1000]:
        filename = 'test'#'H3+-'+str(r) +'.'+str(nshots)+'shots.trial.dat'
        qc_qmc.qc_qmc_ucc(ferm_h, ref, hf_e, nqubits, dt, pop_thresh, filename, nsteps = 2000, shots = nshots, dets=dets, gaussian_noise = False, stoch_hamil=False)
