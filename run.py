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

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("-xyz", "--xyzfile", dest = "xyzfile",
                    help="Molecule .xyz file")
parser.add_argument("-b", "--basis", dest="basis", default='sto-3g', type=str,
                    help="Basis set")
parser.add_argument("-c", "--charge", dest = "charge", default=0, type=int,
                    help="Molecular charge")
parser.add_argument("-m", "--multiplicity", dest = "multiplicity", default=1, type=int,
                    help="Multiplicity")
parser.add_argument("-q", "--nqubits", dest = "nqubits", type=int,
                    help="Number of qubits (should be 2*number of basis states")
parser.add_argument("-e", "--nelectrons", dest = "nelec", type=int,
                    help="Number of electrons (should match charge and multiplicity")
parser.add_argument("-dt", dest="dt", default=0.1, type=float,
                    help="QMC time step")
parser.add_argument("-p", "--pop_threshold", dest="pop_thresh", default=2, type=int,
                    help="Population threshold")
parser.add_argument("-ref", "--reference", dest="ref", type=str, default =None,
                    help="Reference determinant")
parser.add_argument("-dets", "--det_list", dest="dets", type=str, 
                    help="Determinant list file")
parses.add_argument("-shots", dest="nshots", default=0, type=int,
                    help="Number of shots. Default is statevector.")
parses.add_argument("-steps", dest="nsteps", default=1000, type=int,
                    help="Number of QMC steps.")
parser.add_argument("--gaussian", dest="gaussian_noise", action='store_true',
                    help = "Whether to use gaussian noise instead of true shots.")
parser.add_argument("--stoch_hamil", dest="stoch_hamil", action='store_true',
                    help = "Whether to sample the hamiltonian.")
parser.add_argument('-nattempts', dest='nattempts', default=1, type=int,
                    help = "Number of hamiltonian terms to sample beyond diagonal.")
args = parser.parse_args()

if ref == None:
    ref = tuple(x for x in range(nelec))
else:
    ref = tuple(int(x) for x in ref.split(' '))

dets = np.genfromtxt(dets, dtype=int)
for i in range(len(dets)):
    for j in range(nelec):
        dets[i][j] -= 1
dets = [tuple(x) for x in dets]

geometry = utils.get_molecular_geometry(xyzfile) 
hf_e, fci_e, qubit_h = hamil.get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    
ferm_h = normal_ordered(get_fermion_operator(molecule.get_molecular_hamiltonian()))#occupied_indices = [0], active_indices = [1,2,3,4,5,6])))
filename = xyzfile+'_'+basis+'_'+str(charge)+'_'+str(multiplicity)+'_'+str(nshots)+'_shots')
if gaussian_noise:
    filename+='_gaussian'
if stoch_hamil:
    filename+='_'+str(nattempts+1)+'_hamil_groups'
filename+='.dat'

qc_qmc.qc_qmc_ucc(ferm_h, ref, hf_e, nqubits, dt, pop_thresh, filename, nsteps = nsteps, shots = nshots, dets=dets, gaussian_noise = gaussian_noise, stoch_hamil=stoch_hamil, nattempts=nattempts)
