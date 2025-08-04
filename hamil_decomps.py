import numpy as np
import copy
import itertools

from openfermion import jordan_wigner, QubitOperator, FermionOperator
from scipy.sparse import csc_matrix, kron
from scipy.sparse.linalg import spsolve, lsmr
from scipy.linalg import pinv

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
def add_forwards_compatible_paulis(hamil: FermionOperator, num_q: int) -> tuple[list[list[QubitOperator]], dict[tuple, list], list[list[float]]]:
    '''Adds all backwards compatible Pauli strings to sorted insertion collections, splits coefficients evenly, ignores strings with zero variance
    
    Args:
        hamil: fermionic Hamiltonian

    Returns:
        list of collections of mutually commuting Pauli strings with possible repeats, 
        dictionary of operator locations as (collection index, location within collection),
        list of coefficients of Pauli strings by collection
    '''
    coll_list = sorted_insertion(hamil)
    collections_list = copy.deepcopy(coll_list)
    gs_vec = ground_state(hamil, num_q)[0]
    VAR_TOL = 1E-8  # tolerance below whch variance is treated as zero

    # adding all forwards compatible Pauli strings to collection (those not checked in sorted_insertion), splitting coefficients evenly
    checked_ops = set()
    ops_locations = {}  # location of operators
    last_coll_idx = len(collections_list) - 1
    last_coll_locs = {}  # to avoid iterating over all collections again
    for last_coll_op_idx, last_coll_op in enumerate(collections_list[-1]):
        last_coll_locs[list(last_coll_op.terms.keys())[0]] = [(last_coll_idx, last_coll_op_idx)]

    for current_coll_idx, current_coll in enumerate(collections_list[:-1]):  # current collection being checked
        for current_op_idx, current_op in enumerate(current_coll):  # current operator in current collection being checked
            current_op_var = paulis_covariance(current_op, current_op, gs_vec, num_q)
            threshold_var = VAR_TOL * (list(current_op.terms.values())[0]) ** 2
            current_op_key = list(current_op.terms.keys())[0]  # Pauli string only without coefficient
            if current_op_var >= threshold_var:  # only take those strings with nonzero variance up to tolerance
                if current_op_key not in checked_ops:  # have not checked current operator yet
                    op_locs = [(current_coll_idx, current_op_idx)]
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

                    ops_locations[current_op_key] = op_locs
                checked_ops.add(current_op_key)  # avoid checking same operator multiple times
            else:
                ops_locations[current_op_key] = [(current_coll_idx, current_op_idx)]

    ops_locations_dict = ops_locations.copy()
    for op_key, loc in last_coll_locs.items(): 
        if op_key in ops_locations_dict.keys():
            new_loc = copy.deepcopy(ops_locations_dict[op_key])
            new_loc.extend(loc)
            ops_locations_dict[op_key] = new_loc
        else:
            ops_locations_dict[op_key] = loc.copy()
    coeffs_matrix = [[list(op.terms.values())[0] for op in coll] for coll in collections_list]

    return collections_list, ops_locations_dict, coeffs_matrix


def iterative_coefficient_splitting(molecule: str, num_iter: int, ref_collection_method: int, ref_measurement_method: int, ignore_singulars: bool = True) -> tuple[list[list[list[np.float64]]], list[np.ndarray], dict]:
    '''Optimises coefficient split after all backwards compatible Paulis are added to commuting collections (see add_backwards_compatible_paulis)
    
    Args: 
        molecule: molecule name
        num_iter: number of iterations
        ref_collection_method: control how reference collection is selected
            0: random, 1: maximise number of measurements
        ref_measurement_method: control how reference measurement is selected
            0: minimise total measurements and constrain error, 1: minimise error and constrain total measurements
        ignore_singulars: (default True) randomly select a row from the set of its duplicates when solving for coefficients

    Returns:
        out: coefficient matrices for each iteration (which are lists over collections), number of measurements for each iteration, dictionary of operator: list of locations in collections
    '''
    data = get_molecular_data()
    hamil, _, num_q, _ = data[molecule]
    
    collections_list, ops_locations, coeffs_matrix = add_forwards_compatible_paulis(hamil, num_q)
    num_collections = len(collections_list)
    all_coeffs_matrices = []  # list over iterations of coefficient matrices
    all_coeffs_matrices.append(coeffs_matrix)

    # Precomputations
    gs_vec = ground_state(hamil, num_q)[0]
    colls_covariance_mats = collections_paulis_covariance_matrix(collections_list, num_q, gs_vec)  

    initial_collection_stdevs = np.sqrt(np.array(collection_vars_ground_state(hamil, sorted_insertion(hamil), num_q)[1]))
    measurement_nums = np.ceil(initial_collection_stdevs / np.min(initial_collection_stdevs))  # initial number of measurements for each collection
    all_msmt_lists = []  # list over iterations of number of measurements
    all_msmt_lists.append(measurement_nums)

    split_pauli_locs = {}
    coll_wise_single_pauli_locs = {}  # {collection index: {operator: position in collection}} for unsplit Paulis
    coll_wise_split_pauli_locs = {}
    num_split_ops_total = 0
    num_splittings = []
    for pauli, locs in list(ops_locations.items()):
        num_loc = len(locs)
        if num_loc > 1:
            num_split_ops_total += num_loc
            num_splittings.append(num_loc)
            split_pauli_locs[pauli] = locs
            for loc in locs:
                coll_idx, op_idx = loc
                if coll_idx not in coll_wise_split_pauli_locs.keys():
                    coll_wise_split_pauli_locs[coll_idx] = {pauli: op_idx}
                else:
                    coll_wise_split_pauli_locs[coll_idx][pauli] = op_idx
        else:
            coll_idx, op_idx = locs[0]
            if coll_idx not in coll_wise_single_pauli_locs.keys():
                coll_wise_single_pauli_locs[coll_idx] = {pauli: op_idx}
            else:
                coll_wise_single_pauli_locs[coll_idx][pauli] = op_idx

    singular_rows_ops = {}
    for i in range(len(split_pauli_locs)):
        singular_ops = {}
        op = list(split_pauli_locs.keys())[i]
        op_split_colls = [a[0] for a in split_pauli_locs[op]]
        for j in range(len(split_pauli_locs)):
            if i != j:
                check_op = list(split_pauli_locs.keys())[j]
                check_split_colls = [a[0] for a in split_pauli_locs[check_op]]
                if np.array_equal(op_split_colls, check_split_colls):
                    singular_colls = []
                    for k, idx_coll in enumerate(op_split_colls):
                        idx_op = split_pauli_locs[op][k][1]
                        idx_check = split_pauli_locs[check_op][k][1]
                        if np.array_equal(colls_covariance_mats[idx_coll][idx_op], colls_covariance_mats[idx_coll][idx_check]):
                            singular_colls.append(idx_coll)
                    if len(singular_colls) > 0:
                        singular_ops[check_op] = singular_colls
        if len(singular_ops) > 0:
            singular_rows_ops[op] = singular_ops

    # ICS iterations
    iteration = 0
    rng = np.random.default_rng()  # RNG possibly used for picking reference collection(s)
    dvar_list = []
    coll_vars_list = []
    while iteration < num_iter:
        # print(f"BEGINNING ITERATION {iteration + 1}\n")
        new_coeffs_matrix = copy.deepcopy(coeffs_matrix)
        new_measurement_nums = copy.deepcopy(measurement_nums)

        nonsingular_choice_ops = []
        all_singular_ops_lists = []
        for op, singulars in singular_rows_ops.items():
            add_op = True
            for singular_op_list in all_singular_ops_lists:
                if op in singular_op_list:
                    add_op = False
                    break
            if add_op:
                op_singulars_list = [op]
                for singular_op in singulars.keys():
                    op_singulars_list.append(singular_op)
                all_singular_ops_lists.append(op_singulars_list)
                nonsingular_choice_idx = rng.choice(len(op_singulars_list))
                nonsingular_choice_ops.append(op_singulars_list[nonsingular_choice_idx])

        # Update coeffs
        coeffs_opti_matrix = []
        coeffs_opti_res = []
        m_ref: int | float
        solved_op_locs = {}
        for split_op_idx, locs in enumerate(list(split_pauli_locs.values())):
            curr_split_op = list(split_pauli_locs.keys())[split_op_idx]
            if ignore_singulars and (curr_split_op not in singular_rows_ops.keys() or ((curr_split_op in singular_rows_ops.keys()) and (curr_split_op in nonsingular_choice_ops))):
                solved_op_locs[curr_split_op] = locs
                if ref_collection_method == 0:  # random
                    ref_coll_idx, ref_op_idx_in_coll = rng.choice(a=locs)
                    m_ref = measurement_nums[ref_coll_idx]
                elif ref_collection_method == 1:  # maximise number of measurements
                    max_msmt_ref_idx = np.argmax([measurement_nums[idx] for idx in np.array(locs)[:,0]])
                    ref_coll_idx, ref_op_idx_in_coll = locs[max_msmt_ref_idx]
                    m_ref = measurement_nums[ref_coll_idx]

                ref_cov_sum = 0
                for j in coll_wise_single_pauli_locs[ref_coll_idx].values():
                    ref_cov_sum += coeffs_matrix[ref_coll_idx][j] * colls_covariance_mats[ref_coll_idx][ref_op_idx_in_coll][j]
                ref_cov_sum = ref_cov_sum / m_ref

                preserve_sum_equation = np.zeros(num_split_ops_total)
                pre_idx = sum(num_splittings[0:split_op_idx])

                for loc_idx, loc in enumerate(locs):
                    curr_coll_idx, curr_op_idx_in_coll = loc
                    preserve_sum_equation[pre_idx + loc_idx] = 1

                    if curr_coll_idx != ref_coll_idx:
                        covs_equation = np.zeros(num_split_ops_total)
                        m_curr = measurement_nums[curr_coll_idx]
                        curr_cov_sum = 0
                        for j in coll_wise_single_pauli_locs[curr_coll_idx].values():
                            curr_cov_sum += coeffs_matrix[curr_coll_idx][j] * colls_covariance_mats[curr_coll_idx][curr_op_idx_in_coll][j]
                        curr_cov_sum = curr_cov_sum / m_curr
                        res = ref_cov_sum - curr_cov_sum

                        for pstring, i in coll_wise_split_pauli_locs[curr_coll_idx].items():
                            pstring_pre_idx = 0
                            j = 0
                            for check_idx, check_item in enumerate(list(split_pauli_locs.items())):
                                if pstring in check_item:
                                    pstring_pre_idx = sum(num_splittings[0:check_idx])
                                    j = check_item[1].index((curr_coll_idx, i))
                                    break
                            j += pstring_pre_idx
                            covs_equation[j] = coeffs_matrix[curr_coll_idx][i] * colls_covariance_mats[curr_coll_idx][curr_op_idx_in_coll][i] / m_curr
                        
                        for ref_pstring, i in coll_wise_split_pauli_locs[ref_coll_idx].items():
                            ref_pstring_pre_idx = 0
                            j = 0
                            for check_idx, check_item in enumerate(list(split_pauli_locs.items())):
                                if ref_pstring in check_item:
                                    ref_pstring_pre_idx = sum(num_splittings[0:check_idx])
                                    j = check_item[1].index((ref_coll_idx, i))
                                    break
                            j += ref_pstring_pre_idx
                            covs_equation[j] = (-1) * coeffs_matrix[ref_coll_idx][i] * colls_covariance_mats[ref_coll_idx][ref_op_idx_in_coll][i] / m_curr
                        
                        coeffs_opti_matrix.append(covs_equation)
                        coeffs_opti_res.append(res)
                
                current_op_coeff_sum = np.sum([coeffs_matrix[c_idx][o_idx] for c_idx, o_idx in locs])
                coeffs_opti_matrix.append(preserve_sum_equation)
                coeffs_opti_res.append(current_op_coeff_sum)

                new_split_coeffs = lsmr(np.array(coeffs_opti_matrix), coeffs_opti_res, maxiter=2*len(coeffs_opti_res))[0]
        
        for solved_op_idx, locs in enumerate(list(solved_op_locs.values())):
            solved_op = list(solved_op_locs.keys())[solved_op_idx]
            split_op_idx = list(split_pauli_locs.keys()).index(solved_op)
            pre_idx = sum(num_splittings[0:split_op_idx])
            for loc_idx, loc in enumerate(locs):
                new_coeffs_matrix[loc[0]][loc[1]] = new_split_coeffs[pre_idx + loc_idx]

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
        dvar_list.append(dvar)
        coll_vars_list.append(new_colls_vars)

        if ref_measurement_method == 0:  # minimise total number of measurements with dvar constrained
            for mu in range(num_collections):
                new_measurement_nums[mu] = np.ceil(np.sqrt(new_colls_vars[mu]) * new_colls_sum_stdevs / dvar)  # try not rounding up to integer at each iteration
                
        elif ref_measurement_method == 1:  # minimise dvar with total number of measurements constrained
            ref_msmt_coll_idx = np.argmax(new_colls_vars)
            ref_new_coll_var = new_colls_vars[ref_msmt_coll_idx]
            init_msmt_sum = np.sum(measurement_nums)
            msmt_opti_matrix = np.eye(num_collections)
            msmt_opti_res = np.zeros(num_collections)
            for i in range(num_collections):
                if i != ref_msmt_coll_idx:
                    msmt_opti_matrix[i, ref_msmt_coll_idx] = (-1) * np.sqrt(new_colls_vars[i] / ref_new_coll_var)
                else:
                    msmt_opti_matrix[i] = np.ones(num_collections)
                    msmt_opti_res[i] = init_msmt_sum
            
            new_unrounded_msmts = lsmr(msmt_opti_matrix, msmt_opti_res, maxiter=2*num_collections)[0]
            new_measurement_nums = np.round(new_unrounded_msmts, 0)
            sum_diff = init_msmt_sum - np.sum(new_measurement_nums)  # this should only be due to rounding errors
            if sum_diff != 0:
                max_msmt_idx = np.argmax(new_measurement_nums)
                new_measurement_nums[max_msmt_idx] += sum_diff

            for msmt_idx, msmt in enumerate(new_measurement_nums):
                new_max_msmt_idx = np.argmax(new_measurement_nums)
                if msmt <= 0:
                    diff = np.abs(msmt) + 1
                    new_measurement_nums[new_max_msmt_idx] -= diff
                    new_measurement_nums[msmt_idx] = 1

        all_msmt_lists.append(new_measurement_nums)
        measurement_nums = new_measurement_nums
        coeffs_matrix = new_coeffs_matrix

        print(f"ENDING ITERATION {iteration + 1}\n")
        iteration += 1

    return (all_coeffs_matrices, all_msmt_lists, ops_locations, dvar_list, coll_vars_list)


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