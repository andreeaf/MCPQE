import numpy as np
import copy

from functools import reduce
from itertools import combinations
from openfermion import FermionOperator, QubitOperator, MolecularData, qubit_operator_sparse, jordan_wigner, normal_ordered, get_fermion_operator
from openfermionpyscf import run_pyscf
from scipy.sparse import csc_array, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from utils import get_molecular_geometry, dets_list_order

def generate_hf_sd_dets(fermi_hamil: FermionOperator, num_q: int, num_electrons: int,
                        tol: float = 1e-8) -> list[tuple[int]]:
    '''Generate Hartree-Fock reference plus every symmetry-allowed single and double excitation,
    by checking coupling through Hamiltonian matrix.

    Args:
        fermi_hamil: fermionic Hamiltonian of the active space
        num_q: number of spin-orbitals (qubits)
        num_electrons: number of electrons in the active space
        tol: magnitude above which a Hamiltonian matrix element counts as a link

    Returns:
        list of determinants as occupied spin-orbital tuples, reference first
    '''
    ref = tuple(range(num_electrons))
    occupied = list(ref)
    virtual = [p for p in range(num_q) if p not in ref]

    candidates = []
    for i in occupied:                                   # singles, spin conserving
        for a in virtual:
            if (i - a) % 2 == 0:
                candidates.append(tuple(sorted(set(ref) - {i} | {a})))
    for i, j in combinations(occupied, 2):               # doubles, S_z conserving
        for a, b in combinations(virtual, 2):
            if sorted((i % 2, j % 2)) == sorted((a % 2, b % 2)):
                candidates.append(tuple(sorted(set(ref) - {i, j} | {a, b})))

    dets = [ref] + candidates
    idx = [sum(2**(num_q - 1 - p) for p in d) for d in dets]
    hamil_mat = qubit_operator_sparse(jordan_wigner(fermi_hamil), num_q)
    coupling = np.abs(np.asarray(hamil_mat[np.ix_(idx, idx)].todense())) > tol

    reachable, stack = {0}, [0]                          # component containing the reference
    while stack:
        current = stack.pop()
        for other in np.nonzero(coupling[current])[0]:
            if int(other) not in reachable:
                reachable.add(int(other))
                stack.append(int(other))

    return [dets[k] for k in sorted(reachable)]          # index 0 is the reference



def get_molecular_data() -> dict[str, tuple[FermionOperator, list[tuple[int]], int, int]]:
    '''Config data for test molecules

    Returns:
        Dictionary with key = molecule name, value = (fermionic Hamiltonian, allowed dets list, n_qubits, n_electrons, FCI energy, HF energy)
    '''
    config_data = {  # hardcoded config info
    #   name    geometry   allowed dets  occ_idxs  active_idxs    num_qubits  num_electrons
        "h2" : ("h2.xyz" , "h2_dets" ,   None,     None         , 4 ,         2),
        "h4" : ("h4.xyz" , "h4_dets" ,   None,     None         , 8 ,         4),
        "h2o": ("h2o.xyz", "h2o_dets",   [0] ,     [1,2,3,4,5,6], 12,         8),
        "hf" : ("hf.xyz" , "hf_dets" ,   [0] ,     [1,2,3,4,5]  , 10,         8)
    }

    ret_dict = {}
    DATA_FOLDER_PATH = "molecule_data/"
    for key, config_tuple in list(config_data.items()):
        geometry = get_molecular_geometry(DATA_FOLDER_PATH + config_tuple[0])
        molecule = MolecularData(geometry, basis="sto_3g", multiplicity=1, charge=0)
        molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
        fci_energy = molecule.fci_energy
        hf_energy = molecule.hf_energy
        fermi_hamil = normal_ordered(get_fermion_operator(molecule.get_molecular_hamiltonian(config_tuple[2], config_tuple[3])))

        if dets_source == "sd":
            det = generate_hf_sd_dets(fermi_hamil, config_tuple[4], config_tuple[5])
        elif dets_source == "file":
            det = np.genfromtxt(DATA_FOLDER_PATH + config_tuple[1], dtype=int)
            det = det - 1
            det = [tuple(x) for x in det]
        ref = det[0]  # HARDCODED REF: ENSURE REF (HF) IS ALWAYS AT POSITION 0
        det.sort(key=lambda x: dets_list_order(x, ref))  # presort dets in preferred lexicographic order

        ret_dict[key] = (fermi_hamil, det, config_tuple[4], config_tuple[5], fci_energy, hf_energy)

    return ret_dict


def ground_state(hamil: FermionOperator | csc_array | csc_matrix, num_q: int) -> tuple[np.ndarray, float]:
    '''Computes exact ground state of Hamiltonian by expressing it as a sparse matrix
    
    Args:
        hamil: fermionic Hamiltonian
        num_q: number of qubits

    Returns:
        ground state vector, energy
    '''
    if type(hamil) == FermionOperator:
        pauli_hamil = jordan_wigner(hamil)
        pauli_hamil.compress()
        full_hamil_matrix = qubit_operator_sparse(pauli_hamil, num_q).real.astype(float)  # SciPy sparse matrix
    elif type(hamil) == csc_array or csc_matrix:
        full_hamil_matrix = hamil
    else:
        raise TypeError("hamil must be a FermionOperator or scipy.sparse.csc_array")

    ground_state_energy, ground_state_vec = eigsh(full_hamil_matrix, k=1, which='SA')  # smallest eigenvalue (ground state energy)

    return ground_state_vec, ground_state_energy[0]


def collection_hamiltonians(collection_list: list[list[QubitOperator]], num_q: int) -> list[csc_matrix]:
    '''Returns list of Hamiltonian matrices given a list of collections of Pauli strings
    
    Args:
        collection_list: list of all collections
        num_q: number of qubits

    Returns:
        list of Hamiltonians
    '''
    coll_hamils = []
    for collection in collection_list:
        coll_hamil = QubitOperator.zero()
        for op in collection:
            coll_hamil += op
        coll_hamil_mat = qubit_operator_sparse(coll_hamil, num_q)
        coll_hamils.append(coll_hamil_mat.real)  # by Jordan Wigner, Hamiltonian hence collection sub-Hamiltonians must be real

    return coll_hamils


def dets_to_statevector(dets: list[tuple], num_q: int) -> list[np.ndarray]:
    '''Converts determinant tuple (a, b, ...) to statevector of zeroes, 1 at index = 2^a + 2^b + ...
    
    Args:
        dets: list of determinants to convert
        num_q: number of qubits

    Returns:
        statevectors: list of column statevectors
    '''
    statevectors = []
    for det in dets:
        statevector = np.zeros(2**num_q)
        idx = sum([2**orbital for orbital in det])
        statevector[idx] = 1
        statevectors.append(statevector)

    return statevectors


def check_sparse_diagonal(sparse_mat: csc_matrix | csc_array) -> bool:
    '''Checks if Scipy sparse matrix is diagonal
    
    Args:
        sparse_mat: Scipy sparse matrix

    Returns:
        True (if diagonal) or False (if not diagonal)
    '''
    sparse_mat_coo = sparse_mat.tocoo()
    
    return np.all(sparse_mat_coo.row == sparse_mat_coo.col)


def paulis_covariance(op1: QubitOperator, op2: QubitOperator, ref_state: np.ndarray | csc_matrix, num_q: int, normalise: bool = False) -> float:
    '''Returns cov(op1, op2) = <ref_state|op1*op2|ref_state> - <ref_state|op1|ref_state> * <ref_state|op2|ref_state>
    
    Args:
        op1: first qubit operator
        op2: second qubit operator
        ref_state: state on which to compute covariance
        num_q: number of qubits
        normalise: normalise both operators to coefficient 1

    Returns:
        covariance of (op1, op2)
    '''
    if normalise:
        op1 = op1 / list(op1.terms.values())[0]
        op2 = op2 / list(op2.terms.values())[0]
    
    mat1 = qubit_operator_sparse(op1, num_q)
    mat2 = qubit_operator_sparse(op2, num_q)

    if type(ref_state) == np.ndarray:
        ref_state = csc_matrix(ref_state.reshape(-1, 1))

    ref_state_conj = ref_state.getH()
    cov = ref_state_conj @ (mat1 @ mat2) @ ref_state - (ref_state_conj @ mat1 @ ref_state) * (ref_state_conj @ mat2 @ ref_state)

    if type(cov) == csc_matrix or type(cov) == csr_matrix or type(cov) == csc_array:
        cov = cov.data[0] if len(cov.data > 0) else 0

    return cov


def collections_paulis_covariance_matrix(collections_list: list[list[QubitOperator]], num_q: int, ref_vec: np.ndarray) -> list[np.ndarray]:
    '''Computes covariance matrix for Pauli strings (coefficient set to 1) for each collection in list of collections
    
    Args:
        collections_list: list of all collections
        num_q: number of qubits
        ref_vec: reference statevector to calculate covariances with

    Returns:
        list of covariance matrices
    '''
    colls_covariance_mats = []  # list of covariance matrices of each collection
    collections_list_copy = copy.deepcopy(collections_list)

    for collection in collections_list_copy:
        size = len(collection)
        cov_mat = np.zeros(shape=(size,size))
        for op in collection:
            for term in op.terms:
                op.terms[term] = 1  # for covariances we set Pauli string coefficients to 1
        for i in range(size):
            for j in range(i, size):
                op_i = collection[i]
                op_j = collection[j]
                cov_element = paulis_covariance(op_i, op_j, ref_vec, num_q).real

                cov_mat[i, j] = cov_element
                if i != j:
                    cov_mat[j, i] = cov_element
        colls_covariance_mats.append(cov_mat)

    return colls_covariance_mats


def collection_vars_ref_state(collection_list: list[list[QubitOperator]], num_q: int, ref_state: np.ndarray) -> tuple[float, list[float]]:
    '''Variance of each collection of Hamiltonian w.r.t. ref state

    Args:
        collection_list: list of all collections
        num_q: number of qubits
        ref_state: reference state 

    Returns:
        sum of standard deviations, list of variances
    '''
    ref_state_vars = []
    ref_state = csc_matrix(ref_state.reshape(-1, 1))
    coll_hamils = collection_hamiltonians(collection_list, num_q)
    for coll_hamil_mat in coll_hamils:
        var_coll = ((ref_state.getH() @ (coll_hamil_mat @ coll_hamil_mat) @ ref_state) - (ref_state.getH() @ coll_hamil_mat @ ref_state) * (ref_state.getH() @ coll_hamil_mat @ ref_state))
        if type(var_coll) == csc_matrix or type(var_coll) == csr_matrix or type(var_coll) == csc_array:
            var_coll = var_coll.data[0]
        ref_state_vars.append(var_coll.real)

    sum_stdevs = np.sum(np.sqrt(np.array(ref_state_vars)))

    return (sum_stdevs, ref_state_vars)


def powers_of_2(num: int):
    '''Returns tuple of exponents in increasing order (a1, a2, ...) such that num = 2**a1 + 2**a2 + ...
    '''
    check_num = num
    powers = []
    while check_num > 0:
        power = np.floor(np.log2(check_num)).astype(int)
        powers.append(power)
        check_num -= 2 ** power
    powers = tuple(powers[::-1])

    return powers


def flip_binary_bitstring(num: int, num_bits: int) -> int:
    '''Returns the number with flipped binary representation 
    Example: 2 = 10 (binary) -> 01 (binary) = 1
    '''

    bin_num_rev = list(bin(num)[2::][::-1])
    for _ in range(num_bits - len(bin_num_rev)):
        bin_num_rev.append(0)
    rev_num = 0
    for i, j in enumerate(bin_num_rev[::-1]):
        rev_num += int(j) * (2**i)

    return rev_num


# Functions for testing, not directly used in hamil_decomps

x = np.array([[0,1],[1,0]])
z = np.array([[1,0],[0,-1]])
y = np.array([[0, 0-1j], [0+1j, 0]])
id = np.eye(2)
s = np.array([[1, 0], [0, 0+1j]])
def tensorproduct(*args: np.ndarray) -> np.ndarray:
    return reduce(np.kron, args)

def check_mutual_commute(operator_list: list[np.ndarray]) -> bool:
    return all(np.array_equal(op1 @ op2, op2 @ op1) for op1, op2 in combinations(operator_list, 2))
