import numpy as np
import copy

from functools import reduce
from itertools import combinations
from openfermion import FermionOperator, QubitOperator, MolecularData, qubit_operator_sparse, jordan_wigner, normal_ordered, get_fermion_operator
from openfermionpyscf import run_pyscf
from scipy.sparse import csc_array, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from utils import get_molecular_geometry, dets_list_order

def get_molecular_data() -> dict[str, tuple[FermionOperator, list[tuple[int]], int, int]]:
    '''Config data for test molecules

    Returns:
        Dictionary with key = molecule name, value = (fermionic Hamiltonian, allowed dets list, n_qubits, n_electrons)
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
        fermi_hamil = normal_ordered(get_fermion_operator(molecule.get_molecular_hamiltonian(config_tuple[2], config_tuple[3])))

        det = np.genfromtxt(DATA_FOLDER_PATH + config_tuple[1], dtype=int)
        det = det - 1
        det = [tuple(x) for x in det]
        ref = det[0]  # HARDCODED REF: ENSURE REF (HF ANSATZ) IS ALWAYS AT POSITION 0
        det.sort(key=lambda x: dets_list_order(x, ref))  # presort dets in preferred lexicographic order

        ret_dict[key] = (fermi_hamil, det, config_tuple[4], config_tuple[5])

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

    return ground_state_vec, ground_state_energy


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
        ref_state_vars.append(var_coll)

    sum_stdevs = np.sum(np.sqrt(np.array(ref_state_vars)))

    return (sum_stdevs, ref_state_vars)


def ics_bordered_hessian_dets(ops_locations: dict, colls_cov_matrices: list[np.ndarray], measurement_nums: np.ndarray) -> tuple[np.ndarray, str]:
    '''Computes determinants of the (2k+1) to (k+n)th principal leading minors of the ICS Hessian for k distinct operators, total n operators including repeats
    
    Args:
        ops_locations: dictionary of operator locations as operator: list of (collection index, position in collection)
        measurement_nums: list of numbers of measurements of each collection
        colls_cov_matrices: list of covariance matrices of each collection

    Returns:
        list of determinants of principal leading minors,
        "min", "max", or "indeterminate" for local minimum, maximum, or indeterminate respectively
    '''

    distinct_ops = list(ops_locations.keys())
    num_distinct_ops = len(distinct_ops)
    locs = list(ops_locations.values())
    num_ops_in_colls = [len(op_locs) for op_locs in locs]
    num_ops_total = sum(num_ops_in_colls)

    hess_dim = num_distinct_ops + num_ops_total
    upper_zero_submat = np.zeros(shape=(num_distinct_ops, num_distinct_ops))

    lambda_submat = np.zeros(shape=(num_distinct_ops, num_ops_total))
    for op_idx in range(len(distinct_ops)):
        for coll_idx, op_idx_in_coll in locs:
            op_idx_in_all = sum(num_ops_in_colls[0:coll_idx]) + op_idx_in_coll
            lambda_submat[op_idx][op_idx_in_all] = -1

    covs_submats = []
    for coll_idx, coll_measurements in enumerate(measurement_nums):
        covs_submats.append((2 / coll_measurements) * colls_cov_matrices[coll_idx])

    

    return None


# Functions for testing, not directly used in hamil_decomps

# x = np.array([[0,1],[1,0]])
# z = np.array([[1,0],[0,-1]])
# y = np.array([[0, 0-1j], [0+1j, 0]])
# i = np.eye(2)
def tensorproduct(*args: np.ndarray) -> np.ndarray:
    return reduce(np.kron, args)

def check_mutual_commute(operator_list: list[np.ndarray]) -> bool:
    return all(np.array_equal(op1 @ op2, op2 @ op1) for op1, op2 in combinations(operator_list, 2))