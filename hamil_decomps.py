import numpy as np
import copy

from openfermion import jordan_wigner, QubitOperator, FermionOperator
from scipy.sparse import csc_matrix, kron

from hamil_decomp_utils import ground_state, collection_hamiltonians, dets_to_statevector, get_molecular_data, collections_paulis_covariance_matrix, collection_vars_ref_state, paulis_covariance


# --- Testing grouping methods ---
# TODO: REPLACE THESE ARGS WITH SINGLE IDENTIFIER FOR MOLECULE

# 0. Qubitwise Commutativity
def qubitwise_comm(hamil: FermionOperator) -> list[list[QubitOperator]]:
    '''Collects qubitwise commuting terms of Hamiltonian
    
    Args:
        hamil: fermionic Hamiltonian
    
    Returns:
        list of collections of qubitwise commuting Pauli strings
    '''
    pauli_hamil = jordan_wigner(hamil)
    pauli_hamil.compress()
    pauli_hamil_arr = list(pauli_hamil.get_operators())
    collections = {}

    for pauli_string in pauli_hamil_arr:
        op_list = [y for x in list(pauli_string.terms.keys()) for y in x]
        if len(op_list) == 0:  # by default if identity operator exists it is first in pauli_hamil_arr
            collections[()] = [pauli_string]
        no_z_op_list = []  # terms in Pauli string that are not Z
        no_z_op_tuple = ()
        for op in op_list:
            if op[1] != 'Z':
                no_z_op_list.append(op)
            no_z_op_tuple = tuple(no_z_op_list)
        if no_z_op_tuple in list(collections.keys()):
            collections[no_z_op_tuple].append(pauli_string)
        else:
            collections[no_z_op_tuple] = [pauli_string]
        
    # convert collections into list[list[QubitOperator]]
    collections_list = list(collections.values())

    return collections_list


# 1. Sorted Insertion
def sorted_insertion(hamil: FermionOperator) -> list[list[QubitOperator]]:
    '''Collects stringwise commuting terms of Hamiltonian using sorted insertion

    Args:
        fermion_hamil: fermionic Hamiltonian

    Returns:
        list of collections of mutually commuting Pauli strings
    '''
    pauli_hamil = jordan_wigner(hamil)
    pauli_hamil.compress()
    pauli_hamil_arr = list(pauli_hamil.get_operators())

    # Sort Pauli strings by absolute value of coefficients
    sorted_pauli = sorted(
        pauli_hamil_arr,
        key=lambda x: np.abs(list(x.terms.values())[0]),  # x.terms converts QubitOperator to {Pauli string: coefficient}
        reverse=True
    )

    collections_pauli_strings = [] 
    for pauli_string in sorted_pauli:
        for coll in collections_pauli_strings:
            if all(pauli_string * ps == ps * pauli_string for ps in coll):
                coll.append(pauli_string)
                break
        else:  # no collection found whose members all commute with pauli_string
            collections_pauli_strings.append([pauli_string])

    return collections_pauli_strings


# 1.5 Iterative Coefficient Splitting (extension to Sorted Insertion)
def add_backwards_compatible_paulis(hamil: FermionOperator, num_q: int) -> tuple[list[list[QubitOperator]], dict[tuple, list], list[list[float]]]:
    '''Adds all backwards compatible Pauli strings to sorted insertion collections, splits coefficients evenly, ignores strings with zero variance
    
    Args:
        hamil: fermionic Hamiltonian

    Returns:
        list of collections of mutually commuting Pauli strings with possible repeats, 
        dictionary of operator locations as (collection index, location within collection),
        list of coefficients of Pauli strings by collection
    '''
    coll_list = sorted_insertion(hamil)  # function below mutates this directly
    collections_list = copy.deepcopy(coll_list)
    gs_vec = ground_state(hamil, num_q)[0]
    VAR_TOL = 1E-5  # tolerance below whch variance is treated as zero

    # adding all backwards compatible Pauli strings to collection (those not checked in sorted_insertion), splitting coefficients evenly
    checked_ops = set()
    ops_locations = {}  # location of operators
    last_coll_idx = len(collections_list) - 1
    last_coll_locs = {}  # last collection is not accessed below, get its strings' location data now
    for last_coll_op_idx, last_coll_op in enumerate(collections_list[-1]):
        last_coll_locs[list(last_coll_op.terms.keys())[0]] = [(last_coll_idx, last_coll_op_idx)]

    for current_coll_idx, current_coll in enumerate(collections_list[:-1]):  # current collection being checked
        current_coll_coeffs = []
        for current_op_idx, current_op in enumerate(current_coll):  # current operator in current collection being checked
            if paulis_covariance(current_op, current_op, gs_vec, num_q) > VAR_TOL * (list(current_op.terms.values())[0]) ** 2:  # only take those strings with nonzero variance up to tolerance
                current_op_key = list(current_op.terms.keys())[0]  # Pauli string only without coefficient
                op_locs = [(current_coll_idx, current_op_idx)]

                if current_op_key not in checked_ops:  # have not checked current operator yet
                    repeat_idxs = []  # collection indices to add current operator to
                    for check_coll_idx, coll_to_check in enumerate(collections_list[current_coll_idx+1:]):  
                        if all(current_op * ps == ps * current_op for ps in coll_to_check):
                            repeat_coll_idx = check_coll_idx + current_coll_idx + 1
                            repeat_idxs.append(repeat_coll_idx)  # current operator commutes with another collection
                            op_locs.append((repeat_coll_idx, len(coll_to_check)))

                    if len(repeat_idxs) > 0:
                        split_op = current_op / (len(repeat_idxs) + 1)  # evenly split coefficient
                        collections_list[current_coll_idx][current_op_idx] = split_op
                        for repeat_idx in repeat_idxs:
                            collections_list[repeat_idx].append(split_op)

                current_coll_coeffs.append(current_op.terms[current_op_key])
                checked_ops.add(current_op_key)  # avoid checking same operator multiple times
                ops_locations[current_op_key] = op_locs
        
    ops_locations_dict = ops_locations | last_coll_locs
    coeffs_matrix = [[list(op.terms.values())[0] for op in coll] for coll in collections_list]

    return collections_list, ops_locations_dict, coeffs_matrix

def iterative_coefficient_splitting(molecule: str, num_iter: int) -> tuple[list[list[list[np.float64]]], list[np.ndarray], dict]:
    '''Optimises coefficient split after all backwards compatible Paulis are added to commuting collections (see add_backwards_compatible_paulis)
    
    Args: 
        molecule: molecule to perform ICS on
        num_iter: number of iterations

    Returns:
        out: coefficient matrices for each iteration (which are lists over collections), number of measurements for each iteration, dictionary of operator: list of locations in collections
    '''
    data = get_molecular_data()
    hamil, _, num_q, _ = data[molecule]
    
    collections_list, ops_locations, coeffs_matrix = add_backwards_compatible_paulis(hamil, num_q)
    all_coeffs_matrices = []  # list over iterations of coefficient matrices
    all_coeffs_matrices.append(coeffs_matrix)


    # Precomputations
    gs_vec = ground_state(hamil, num_q)[0]  # NOTE: this is only for testing, change this to current wavefunction
    colls_covariance_mats = collections_paulis_covariance_matrix(collections_list, num_q, gs_vec)  

    initial_collection_stdevs = np.sqrt(np.array(collection_vars_ground_state(hamil, sorted_insertion(hamil), num_q)[1]))
    measurement_nums = np.ceil(initial_collection_stdevs / np.min(initial_collection_stdevs))  # initial number of measurements for each collection
    all_msmt_lists = []  # list over iterations of number of measurements
    all_msmt_lists.append(measurement_nums)

    split_pauli_locs = {}  # locations of split paulis, we obviously only compute coefficient splitting for these
    for pauli, loc in list(ops_locations.items()):
        if len(loc) > 1:
            split_pauli_locs[pauli] = loc

    # ICS iterations
    iteration = 0
    rng = np.random.default_rng()  # RNG possibly used for picking reference collection(s)
    while iteration < num_iter:
        new_coeffs_matrix = copy.deepcopy(coeffs_matrix)
        new_measurement_nums = copy.deepcopy(measurement_nums)

        # Update coeffs
        for _, locs in split_pauli_locs.items():  # locs is an array of 2-tuples: (collection index, position index in collection) for P_k (current Pauli)

            # Picking reference collection
            # 1. Random (possibly with nonuniform distribution)
            # ref_coll_idx, ref_op_idx_in_coll = rng.choice(a=locs)  # random collection

            # 2. Maxmimise number of measurements
            max_msmt_ref_idx = np.argmax([measurement_nums[idx] for idx in np.array(locs)[:,0]])  # collection that maximises number of measurements
            ref_coll_idx, ref_op_idx_in_coll = locs[max_msmt_ref_idx]
            

            m_ref = measurement_nums[ref_coll_idx]
            ref_covs = []
            for j in range(len(coeffs_matrix[ref_coll_idx])):
                cov_row = np.min((j, ref_op_idx_in_coll))  # covariance matrix is upper triangular
                cov_col = np.max((j, ref_op_idx_in_coll))
                ref_covs.append(coeffs_matrix[ref_coll_idx][j] * colls_covariance_mats[ref_coll_idx][cov_row][cov_col])  
            ref_cov_sum = np.sum(ref_covs)  # sum over ref collection of c_i^ref * cov(P_k, P_i)
            
            op_var = colls_covariance_mats[ref_coll_idx][ref_op_idx_in_coll][ref_op_idx_in_coll]  # var(P_k)
            current_op_coeff_sum = 0  # sum of coefficients of P_k (will remain constant)
            new_non_ref_coeff_sum = 0  # sum of non-reference-collection coefficients of P_k

            for loc in locs:
                curr_coll_idx, curr_op_idx_in_coll = loc  # collection whose coefficient is being changed
                current_op_coeff_sum += coeffs_matrix[curr_coll_idx][curr_op_idx_in_coll]
                if curr_coll_idx != ref_coll_idx:  # not reference collection
                    curr_cov_sum = 0  # sum over j != k of c_j^nu * cov(P_k, P_j)
                    for j in range(len(coeffs_matrix[curr_coll_idx])):
                        if j != curr_op_idx_in_coll:
                            cov_row = np.min((j, curr_op_idx_in_coll))  # covariance matrix is upper triangular
                            cov_col = np.min((j, curr_op_idx_in_coll))
                            curr_cov_sum += coeffs_matrix[curr_coll_idx][j] * colls_covariance_mats[curr_coll_idx][cov_row][cov_col]
                    new_coeff = (1/op_var) * ((measurement_nums[curr_coll_idx] / m_ref)*ref_cov_sum - curr_cov_sum)
                    new_non_ref_coeff_sum += new_coeff
                    new_coeffs_matrix[curr_coll_idx][curr_op_idx_in_coll] = new_coeff

            new_coeffs_matrix[ref_coll_idx][ref_op_idx_in_coll] = current_op_coeff_sum - new_non_ref_coeff_sum  # ensure total coefficient sum stays constant

        all_coeffs_matrices.append(new_coeffs_matrix)

        # Update measurement numbers 
        new_collections_list = []  # collections with new coefficients
        for coll_idx, coll in enumerate(collections_list):
            new_coll = []
            for op_idx_in_coll, op in enumerate(coll):
                new_op = copy.deepcopy(op)
                for term in new_op.terms:
                    new_op.terms[term] = new_coeffs_matrix[coll_idx][op_idx_in_coll]
                new_coll.append(new_op)
            new_collections_list.append(new_coll)
        
        new_colls_sum_stdevs, new_colls_vars = collection_vars_ref_state(new_collections_list, num_q, ref_state=gs_vec)  # sum over collections of collection standard deviations, list of collection variances
        dvar = np.sum([new_colls_var / measurement_nums[mu] for mu, new_colls_var in enumerate(new_colls_vars)])

        # 1. Minimise total number of measurements with dvar constrained
        for mu in range(len(measurement_nums)):
            new_measurement_nums[mu] = np.sqrt(new_colls_vars[mu]) * new_colls_sum_stdevs / dvar  # try not rounding up to integer at each iteration

        # 2. Minimise dvar with total number of measurements constrained
        # ref_msmt_coll_idx = np.argmax(new_colls_vars)
        # ref_msmt = measurement_nums[ref_msmt_coll_idx]
        # ref_new_coll_var = new_colls_vars[ref_msmt_coll_idx]
        # for mu in range(len(measurement_nums)):
        #     new_measurement_nums[mu] = np.ceil(ref_msmt * np.sqrt(new_colls_vars[mu] / ref_new_coll_var))

        all_msmt_lists.append(new_measurement_nums)
        measurement_nums = new_measurement_nums
        coeffs_matrix = new_coeffs_matrix

        iteration += 1

    return (all_coeffs_matrices, all_msmt_lists, ops_locations)


# --- Metrics for grouping methods ---

# 1. Collection matrix variance using exact ground state
def collection_vars_ground_state(hamil: FermionOperator, collection_list: list[list[QubitOperator]], num_q: int) -> tuple[float, list[float]]:
    '''Variance of each collection of Hamiltonian w.r.t. exact ground state

    Args:
        hamil: fermionic Hamiltonian
        collection_list: list of all collections
        num_q: number of qubits

    Returns:
        sum of standard deviations, list of variances
    '''
    ground_state_vars = []
    ground_statevec = ground_state(hamil, num_q)[0]
    ground_statevec_conj = ground_statevec.conj().T
    coll_hamils = collection_hamiltonians(collection_list, num_q)
    for coll_hamil_mat in coll_hamils:
        var_coll = ((ground_statevec_conj @ (coll_hamil_mat @ coll_hamil_mat) @ ground_statevec) - (ground_statevec_conj @ coll_hamil_mat @ ground_statevec) ** 2)[0][0]
        ground_state_vars.append(var_coll)

    sum_stdevs = np.sum(np.sqrt(np.array(ground_state_vars)))

    return (sum_stdevs, ground_state_vars)


# 2. "Residual variance" using modified Hadamard test
def hadamard_test_residual_variance(hamil: FermionOperator, collection_list: list[list[QubitOperator]], num_q: int, dets: list[tuple[int]], ref_idx: int = 0) -> tuple[float, float]:
    '''Measures residual variance for each term in Hamiltonian decomposition using Hadamard test circuit and exact ground state
    
    Args:
        hamil: fermionic Hamiltonian
        collection_list: list of all collections
        num_q: number of qubits
        dets: list of allowed determinants (including HF ansatz == reference)
        ref_idx: index in dets of ansatz (reference), default = 0

    Returns:
        (sum of standard deviations, sum of variances)
    '''
    dim = 2**num_q
    ground_state_vec = ground_state(hamil, num_q)[0]
    ground_state_vec = ground_state_vec.flatten()
    coll_hamils = collection_hamiltonians(collection_list, num_q)
    det_vecs = dets_to_statevector(dets, num_q)
    del det_vecs[ref_idx]

    pauli_z = np.array([[1, 0], [0, -1]])
    sparse_pauli_z = csc_matrix(pauli_z)

    var_list = []
    for det_vec in det_vecs:
        coll_vars = []
        for coll_hamil in coll_hamils:
            init_state = 0.5 * np.hstack((np.zeros(dim), det_vec + ground_state_vec)) + 0.5 * np.hstack((det_vec - ground_state_vec, np.zeros(dim)))
            sparse_init_state = csc_matrix(init_state.reshape(-1,1))
            circ_mat = kron(sparse_pauli_z, coll_hamil)  # scipy.sparse.kron, NOT np.kron
            sparse_init_state_conj = sparse_init_state.getH()
            coll_var = sparse_init_state_conj @ circ_mat @ circ_mat @ sparse_init_state - (sparse_init_state_conj @ circ_mat @ sparse_init_state) ** 2
            coll_vars.append(coll_var.data[0])

        var_list.append(coll_vars)

    var_list = np.array(var_list)
    stdev_list = np.sqrt(var_list)
    summed_vars = np.sum(var_list, axis=1)
    summed_stdevs = np.sum(stdev_list, axis=1)

    return (np.sum(summed_stdevs), np.sum(summed_vars))