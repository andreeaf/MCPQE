import numpy as np
import copy
import itertools

from openfermion import jordan_wigner, QubitOperator, FermionOperator, qubit_operator_sparse
from scipy.sparse import csc_matrix, kron, identity
from scipy.sparse.linalg import spsolve, lsmr

from hamil_decomp_utils import ground_state, collection_hamiltonians, dets_to_statevector, get_molecular_data, collections_paulis_covariance_matrix, collection_vars_ref_state, paulis_covariance, powers_of_2, flip_binary_bitstring


# --- Testing grouping methods ---
# TODO: REPLACE THESE ARGS WITH SINGLE IDENTIFIER FOR MOLECULE

# 0. Qubitwise Commutativity
def qubitwise_comm(molecule: str) -> list[list[QubitOperator]]:
    '''Collects qubitwise commuting terms of Hamiltonian
    
    Args:
        molecule: name of molecule
    
    Returns:
        list of collections of qubitwise commuting Pauli strings
    '''
    data = get_molecular_data()
    hamil = data[molecule][0]
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
def sorted_insertion(molecule: str = None, fermi_hamil: FermionOperator = None) -> list[list[QubitOperator]]:
    '''Collects stringwise commuting terms of Hamiltonian using sorted insertion

    Args:
        molecule: name of molecule
        fermi_hamil: fermionic Hamiltonian (will not be used if molecule is given)

    Returns:
        list of collections of mutually commuting Pauli strings
    '''
    if qubit_hamil is not None:
        pauli_hamil = qubit_hamil
    else:
        if molecule:
            data = get_molecular_data()
            hamil = data[molecule][0]
        else:
            hamil = fermi_hamil

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
def add_forwards_compatible_paulis(molecule: str) -> tuple[list[list[QubitOperator]], dict[tuple, list], list[list[float]]]:
    '''Adds all backwards compatible Pauli strings to sorted insertion collections, splits coefficients evenly, ignores strings with zero variance
    
    Args:
        molecule: name of molecule

    Returns:
        list of collections of mutually commuting Pauli strings with possible repeats, 
        dictionary of operator locations as (collection index, location within collection),
        list of coefficients of Pauli strings by collection
    '''
    data = get_molecular_data()
    hamil, _, num_q, _, _, _ = data[molecule]
    coll_list = sorted_insertion(molecule)
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


def iterative_coefficient_splitting_naive(molecule: str, 
                                          num_meas: int, 
                                          num_iter: int) -> tuple[list[list[list[np.float64]]], list[list[int]]]:
    data = get_molecular_data()
    hamil, flipped_dets, num_q, _, _, _ = data[molecule]
    dim = 2**num_q
    pauli_z = np.array([[1, 0], [0, -1]])
    iden = np.array([[1, 0], [0, 1]])
    SPARSE_Z = csc_matrix(pauli_z)
    SPARSE_I = csc_matrix(iden)
    HF_DET_IDX = 0  # Hartree-Fock determinant index in flipped_dets, default 0
    M = num_meas
    NUM_ITER = num_iter

    dets = [powers_of_2(
            flip_binary_bitstring(
                sum([2**k for k in flipped_det]),
                num_q)) for flipped_det in flipped_dets
    ]
    dets.pop(HF_DET_IDX) 
    det_idxs = list(map(lambda x: sum([2**k for k in x]), dets))
    det_vecs = dets_to_statevector(dets, num_q)

    collections_list, ops_locations, coeffs_matrix = add_forwards_compatible_paulis(molecule)
    # num_collections = len(collections_list)
    init_coeffs = {}
    for coll in collections_list:
        for op in coll:
            op_key = list(op.terms.keys())[0]
            op_val = list(op.terms.values())[0]
            if op_key in init_coeffs:
                init_coeffs[op_key] += op_val
            else:
                init_coeffs[op_key] = op_val
    all_coeffs_matrices = [] 
    all_coeffs_matrices.append(coeffs_matrix)

    norm_colls_list = [
        [QubitOperator(list(op.terms.keys())[0]) for op in coll]
        for coll in collections_list
    ]  # collections with all operator coefficients normalised to 1
    coll_op_idx_pairs = [
        list(itertools.combinations(range(len(coll)), 2)) for coll in collections_list
    ]  # list of all distinct pairs of operator indices for each collection
    coll_op_pairs = [
        [(norm_colls_list[coll_idx][idx_pair[0]], norm_colls_list[coll_idx][idx_pair[1]]) for idx_pair in coll] 
        for coll_idx, coll in enumerate(coll_op_idx_pairs)
    ]  # list of all distinct pairs of operators for each collection

    op_pair_prods = {}
    op_single_prods = {}
    for coll in coll_op_pairs:
        for op1, op2 in coll:
            op1_key = list(op1.terms.keys())[0]
            op2_key = list(op2.terms.keys())[0]
            if ((op1_key, op2_key) not in op_pair_prods) and ((op2_key, op1_key) not in op_pair_prods):
                op1_mat = qubit_operator_sparse(op1, num_q)
                op2_mat = qubit_operator_sparse(op2, num_q)
                prod = kron(SPARSE_I, op1_mat @ op2_mat)
                op_pair_prods[(op1_key, op2_key)] = prod

    for coll in norm_colls_list:
        for op in coll:
            op_key = list(op.terms.keys())[0]
            if op_key not in op_single_prods:
                op_mat = qubit_operator_sparse(op, num_q)
                prod = kron(SPARSE_Z, op_mat)
                op_single_prods[op_key] = prod

    gs_vec, _ = ground_state(hamil, num_q)
    gs_vec = gs_vec.flatten()
    resvar_init_states = {}
    sparse_resvar_init_states = {}  # sparse init (ket, bra) vectors
    for det_idx, det in enumerate(dets):
        det_vec = det_vecs[det_idx]
        init_statevec = 0.5*np.hstack((np.zeros(dim), det_vec+gs_vec)) + 0.5*np.hstack((det_vec-gs_vec, np.zeros(dim)))
        resvar_init_states[det] = init_statevec
        sparse_resvar_init_states[det] = (csc_matrix(np.conj(init_statevec)), csc_matrix(init_statevec.reshape(-1,1)))

    NUM_DETS = len(dets)

    op_single_prods_all_dets = {
        op_key: np.array([
            (ket_vec @ prod_mat @ bra_vec).data[0].real for ket_vec, bra_vec in sparse_resvar_init_states.values()
        ]) 
        for op_key, prod_mat in op_single_prods.items()
    }

    op_double_prods_all_dets = {
        op_key_pair: sum([
            (ket_vec @ prod_mat @ bra_vec).data[0].real for ket_vec, bra_vec in sparse_resvar_init_states.values()
        ])
        for op_key_pair, prod_mat in op_pair_prods.items()
    }
            
    # colls_covariance_mats = collections_paulis_covariance_matrix(collections_list, num_q, gs_vec)  
    # initial_collection_variances = []
    # for i, coll in enumerate(collections_list):
    #     var = 0
    #     for j in range(len(coll)):
    #         for k in range(len(coll)):
    #             var += colls_covariance_mats[i][j,k]*coeffs_matrix[i][j]*coeffs_matrix[i][k]
    #     initial_collection_variances.append(var)
    # varH = (sum(np.sqrt(x) for x in initial_collection_variances))**2/M
    # measurement_nums = [np.ceil(np.sqrt(x) * sum(np.sqrt(y) for y in initial_collection_variances)/varH) for x in initial_collection_variances]

    _, _, resvars = hadamard_test_residual_variance(molecule, collections_list)
    resvars_by_colls = np.sum(resvars, axis=0)
    measurement_nums = np.ceil((M/np.sum(np.sqrt(resvars_by_colls))) * np.sqrt(resvars_by_colls)).tolist()
    all_msmt_lists = []  
    all_msmt_lists.append(measurement_nums)
    num_params = sum(len(x) for x in collections_list)
    ref_colls = {op: ops_locations[op][0] for op in ops_locations}
    # print(-1, varH, measurement_nums)
    for it in range(NUM_ITER):
        if it == 0 or (it + 1) % 5 == 0:
            print(f"ITERATION {it+1}")
        matrix = np.zeros((num_params, num_params))
        result = np.zeros(num_params)
        meas = all_msmt_lists[-1]

        #Compute matrix and result vector for linear equations
        # for index, coll in enumerate(collections_list):
        #     for j, op in enumerate(coll):
        #         ind1 = sum(len(x) for x in collections_list[:index]) + j
        #         op_key = list(op.terms.keys())[0]
        #         if index != ref_colls[op_key][0]:
        #             for k, op2 in enumerate(coll):
        #                 ind2 = sum(len(x) for x in collections_list[:index]) + k
        #                 matrix[ind1, ind2] = colls_covariance_mats[index][j,k]/meas[index]        
        #             for k, op2 in enumerate(collections_list[ref_colls[op_key][0]]):
        #                 ind2 = sum(len(x) for x in collections_list[:ref_colls[op_key][0]]) + k
        #                 matrix[ind1, ind2] = -colls_covariance_mats[ref_colls[op_key][0]][ref_colls[op_key][1],k]/meas[ref_colls[op_key][0]]        
        #         else:
        #             for loc in ops_locations[op_key]:
        #                 matrix[ind1][sum(len(x) for x in collections_list[:loc[0]]) + loc[1]] = 1
        #                 result[ind1] = init_coeffs[op_key]

        for coll_idx, coll in enumerate(collections_list):
            idx_of_coll_start = sum(len(x) for x in collections_list[:coll_idx])
            m_coll = meas[coll_idx]
            for op_idx_1, op1 in enumerate(coll):
                idx1 = idx_of_coll_start + op_idx_1
                op1_key = list(op1.terms.keys())[0]
                ref_coll_idx = ref_colls[op1_key][0]
                m_ref = meas[ref_coll_idx]
                if coll_idx != ref_coll_idx:
                    for op_idx_2, op2 in enumerate(coll):
                        idx2 = idx_of_coll_start + op_idx_2
                        op2_key = list(op2.terms.keys())[0]
                        op_single_term = op_single_prods_all_dets[op1_key] @ op_single_prods_all_dets[op2_key]
                        if op_idx_1 == op_idx_2:
                            matrix[idx1, idx2] = NUM_DETS - op_single_term
                        else:
                            possible_op_pairs = ((op1_key, op2_key), (op2_key, op1_key))
                            op_keys_pair = possible_op_pairs[0] if possible_op_pairs[0] in op_double_prods_all_dets.keys() else possible_op_pairs[1]
                            matrix[idx1, idx2] = op_double_prods_all_dets[op_keys_pair] - op_single_term
                    
                    for ref_op_idx, ref_op in enumerate(collections_list[ref_coll_idx]):
                        ref_op_key = list(ref_op.terms.keys())[0]
                        idx_ref = sum(len(x) for x in collections_list[:ref_coll_idx]) + ref_op_idx
                        m_ratio = m_coll / m_ref
                        ref_op_single_term = op_single_prods_all_dets[op1_key] @ op_single_prods_all_dets[ref_op_key]
                        if ref_op_key == op1_key:
                            matrix[idx1, idx_ref] = -m_ratio * (NUM_DETS - ref_op_single_term)
                        else:
                            possible_op_pairs = ((op1_key, ref_op_key), (ref_op_key, op1_key))
                            op_keys_pair = possible_op_pairs[0] if possible_op_pairs[0] in op_double_prods_all_dets.keys() else possible_op_pairs[1]
                            matrix[idx1, idx_ref] = -m_ratio * (op_double_prods_all_dets[op_keys_pair] - ref_op_single_term)

                else:
                    for loc in ops_locations[op1_key]:
                        matrix[idx1][sum(len(x) for x in collections_list[:loc[0]]) + loc[1]] = 1
                        result[idx1] = init_coeffs[op1_key]

        #Solve linear equations and store result
        mat_inv = np.linalg.pinv(matrix)
        coeffs = mat_inv.dot(result)
        coeffs_matrix = copy.deepcopy(all_coeffs_matrices[-1])
        for i, coll in enumerate(coeffs_matrix):
            for j in range(len(coll)):
                ind1 = sum(len(x) for x in collections_list[:i]) + j
                coll[j] = coeffs[ind1]
        all_coeffs_matrices.append(coeffs_matrix)

        #Compute variance and measurement numbers
        # collection_variances = []
        # for i, coll in enumerate(collections_list):
        #     var = 0
        #     for j in range(len(coll)):
        #         for k in range(len(coll)):
        #             var += colls_covariance_mats[i][j,k]*coeffs_matrix[i][j]*coeffs_matrix[i][k]
        #     collection_variances.append(var)
        # # varH = (sum(np.sqrt(x) for x in collection_variances))**2/M
        # # measurement_nums = [np.ceil(np.sqrt(x) * sum(np.sqrt(y) for y in collection_variances)/varH) for x in collection_variances]
        # # all_msmt_lists.append(measurement_nums)
        # measurement_nums = [np.ceil((M / np.sum([np.sqrt(y) for y in collection_variances])) * np.sqrt(x)) for x in collection_variances]

        curr_colls = copy.deepcopy(collections_list)
        for coll_idx, coll_coeffs in enumerate(coeffs_matrix):
            for op_idx, coeff in enumerate(coll_coeffs):
                op_key = list(curr_colls[coll_idx][op_idx].terms.keys())[0]
                curr_colls[coll_idx][op_idx].terms[op_key] = coeff

        _, _, resvars = hadamard_test_residual_variance(molecule, curr_colls)
        sqrt_resvars_by_coll = np.sqrt(np.sum(resvars, axis=0))
        measurement_nums = np.ceil(M * sqrt_resvars_by_coll / np.sum(sqrt_resvars_by_coll))

        all_msmt_lists.append(measurement_nums)
        
        
    return all_coeffs_matrices, all_msmt_lists


def iterative_coefficient_splitting_penalty(molecule: str, 
                                          num_meas: int, 
                                          num_iter: int,
                                          penalty_coeffs: list[int]) -> tuple[list[list[list[np.float64]]], list[list[int]]]:
    data = get_molecular_data()
    hamil, flipped_dets, num_q, _, _, _ = data[molecule]
    dim = 2**num_q
    pauli_z = np.array([[1, 0], [0, -1]])
    iden = np.array([[1, 0], [0, 1]])
    SPARSE_Z = csc_matrix(pauli_z)
    SPARSE_I = csc_matrix(iden)
    HF_DET_IDX = 0  # Hartree-Fock determinant index in flipped_dets, default 0
    M = num_meas
    NUM_ITER = num_iter
    PENALTY_COEFFS = penalty_coeffs  # sequentially applied penalty coefficients

    dets = [powers_of_2(
            flip_binary_bitstring(
                sum([2**k for k in flipped_det]),
                num_q)) for flipped_det in flipped_dets
    ]
    dets.pop(HF_DET_IDX) 
    det_vecs = dets_to_statevector(dets, num_q)

    collections_list, ops_locations, coeffs_matrix = add_forwards_compatible_paulis(molecule)
    # num_collections = len(collections_list)
    init_coeffs = {}
    for coll in collections_list:
        for op in coll:
            op_key = list(op.terms.keys())[0]
            op_val = list(op.terms.values())[0]
            if op_key in init_coeffs:
                init_coeffs[op_key] += op_val
            else:
                init_coeffs[op_key] = op_val
    all_coeffs_matrices = [] 
    all_coeffs_matrices.append(coeffs_matrix)

    norm_colls_list = [
        [QubitOperator(list(op.terms.keys())[0]) for op in coll]
        for coll in collections_list
    ]  # collections with all operator coefficients normalised to 1
    coll_op_idx_pairs = [
        list(itertools.combinations(range(len(coll)), 2)) for coll in collections_list
    ]  # list of all distinct pairs of operator indices for each collection
    coll_op_pairs = [
        [(norm_colls_list[coll_idx][idx_pair[0]], norm_colls_list[coll_idx][idx_pair[1]]) for idx_pair in coll] 
        for coll_idx, coll in enumerate(coll_op_idx_pairs)
    ]  # list of all distinct pairs of operators for each collection

    op_pair_prods = {}
    op_single_prods = {}
    for coll in coll_op_pairs:
        for op1, op2 in coll:
            op1_key = list(op1.terms.keys())[0]
            op2_key = list(op2.terms.keys())[0]
            if ((op1_key, op2_key) not in op_pair_prods) and ((op2_key, op1_key) not in op_pair_prods):
                op1_mat = qubit_operator_sparse(op1, num_q)
                op2_mat = qubit_operator_sparse(op2, num_q)
                prod = kron(SPARSE_I, op1_mat @ op2_mat)
                op_pair_prods[(op1_key, op2_key)] = prod

    for coll in norm_colls_list:
        for op in coll:
            op_key = list(op.terms.keys())[0]
            if op_key not in op_single_prods:
                op_mat = qubit_operator_sparse(op, num_q)
                prod = kron(SPARSE_Z, op_mat)
                op_single_prods[op_key] = prod

    gs_vec, _ = ground_state(hamil, num_q)
    gs_vec = gs_vec.flatten()
    resvar_init_states = {}
    sparse_resvar_init_states = {}  # sparse init (ket, bra) vectors
    for det_idx, det in enumerate(dets):
        det_vec = det_vecs[det_idx]
        init_statevec = 0.5*np.hstack((np.zeros(dim), det_vec+gs_vec)) + 0.5*np.hstack((det_vec-gs_vec, np.zeros(dim)))
        resvar_init_states[det] = init_statevec
        sparse_resvar_init_states[det] = (csc_matrix(np.conj(init_statevec)), csc_matrix(init_statevec.reshape(-1,1)))

    NUM_DETS = len(dets)

    op_single_prods_all_dets = {
        op_key: np.array([
            (ket_vec @ prod_mat @ bra_vec).data[0].real for ket_vec, bra_vec in sparse_resvar_init_states.values()
        ]) 
        for op_key, prod_mat in op_single_prods.items()
    }

    op_double_prods_all_dets = {
        op_key_pair: sum([
            (ket_vec @ prod_mat @ bra_vec).data[0].real for ket_vec, bra_vec in sparse_resvar_init_states.values()
        ])
        for op_key_pair, prod_mat in op_pair_prods.items()
    }

    _, _, resvars = hadamard_test_residual_variance(molecule, collections_list)
    resvars_by_colls = np.sum(resvars, axis=0)
    measurement_nums = np.ceil((M/np.sum(np.sqrt(resvars_by_colls))) * np.sqrt(resvars_by_colls)).tolist()
    all_msmt_lists = []  
    all_msmt_lists.append(measurement_nums)
    num_params = sum(len(x) for x in collections_list)
    for p_coeff in PENALTY_COEFFS:
        print(f"PENALTY COEFFICIENT: {p_coeff}")
        for it in range(NUM_ITER):
            if it == 0 or (it + 1) % 5 == 0:
                print(f"ITERATION {it+1}")
            matrix = np.zeros((num_params, num_params))
            result = np.zeros(num_params)
            meas = all_msmt_lists[-1]

            for coll_idx, coll in enumerate(collections_list):
                idx_of_coll_start = sum(len(x) for x in collections_list[:coll_idx])
                m_coll = meas[coll_idx]
                for op_idx_1, op1 in enumerate(coll):
                    idx1 = idx_of_coll_start + op_idx_1
                    op1_key = list(op1.terms.keys())[0]
                    op_locs = ops_locations[op1_key]

                    for op_idx_2, op2 in enumerate(coll):
                        idx2 = idx_of_coll_start + op_idx_2
                        op2_key = list(op2.terms.keys())[0]
                        op_single_term = op_single_prods_all_dets[op1_key] @ op_single_prods_all_dets[op2_key]
                        if op_idx_1 == op_idx_2:
                            matrix[idx1, idx2] = (2/m_coll) * (NUM_DETS - op_single_term)
                        else:
                            possible_op_pairs = ((op1_key, op2_key), (op2_key, op1_key))
                            op_keys_pair = possible_op_pairs[0] if possible_op_pairs[0] in op_double_prods_all_dets.keys() else possible_op_pairs[1]
                            matrix[idx1, idx2] = (2/m_coll) * (op_double_prods_all_dets[op_keys_pair] - op_single_term)

                    for loc_coll_idx, loc_op_idx in op_locs:
                        loc_idx = sum(len(x) for x in collections_list[:loc_coll_idx]) + loc_op_idx
                        matrix[idx1, loc_idx] = 2 * p_coeff 

                    result[idx1] = 2 * p_coeff * init_coeffs[op1_key]

            #Solve linear equations and store result
            mat_inv = np.linalg.pinv(matrix)
            coeffs = mat_inv.dot(result)
            coeffs_matrix = copy.deepcopy(all_coeffs_matrices[-1])
            for i, coll in enumerate(coeffs_matrix):
                for j in range(len(coll)):
                    ind1 = sum(len(x) for x in collections_list[:i]) + j
                    coll[j] = coeffs[ind1]
            all_coeffs_matrices.append(coeffs_matrix)

            #Compute variance and measurement numbers
            curr_colls = copy.deepcopy(collections_list)
            for coll_idx, coll_coeffs in enumerate(coeffs_matrix):
                for op_idx, coeff in enumerate(coll_coeffs):
                    op_key = list(curr_colls[coll_idx][op_idx].terms.keys())[0]
                    curr_colls[coll_idx][op_idx].terms[op_key] = coeff

            _, _, resvars = hadamard_test_residual_variance(molecule, curr_colls)
            sqrt_resvars_by_coll = np.sqrt(np.sum(resvars, axis=0))
            measurement_nums = np.ceil(M * sqrt_resvars_by_coll / np.sum(sqrt_resvars_by_coll))

            all_msmt_lists.append(measurement_nums)
        
        
    return all_coeffs_matrices, all_msmt_lists


def total_resvars_by_iteration(molecule, coeffs_mats, msmts_lists, init_colls):
    total_resvars = []
    for iter_idx, coeffs_mat in enumerate(coeffs_mats):
        check_colls = copy.deepcopy(init_colls)
        for coll_idx, coll_coeffs in enumerate(coeffs_mat):
            for op_idx, coeff in enumerate(coll_coeffs):
                op_key = list(check_colls[coll_idx][op_idx].terms.keys())[0]
                check_colls[coll_idx][op_idx].terms[op_key] = coeff

        _, _, c = hadamard_test_residual_variance(molecule, check_colls)
        
        total_resvar = np.sum(
            np.sum(c, axis=0) / msmts_lists[iter_idx]
        )
        print(total_resvar)
        total_resvars.append(total_resvar)
    return total_resvars

def op_coeff_diff(ops_locs: dict, coeffs_mats):
    diff_dict = {
        op: sum(coeffs_mats[-1][i][j] for i,j in locs) - sum(coeffs_mats[0][i][j] for i,j in locs)
        for op, locs in ops_locs.items() 
    }
    
    return diff_dict


# --- Metrics for grouping methods ---

# 1. Collection matrix variance using exact ground state
def collection_vars_ground_state(molecule: str, collection_list: list[list[QubitOperator]]) -> tuple[float, list[float]]:
    '''Variance of each collection of Hamiltonian w.r.t. exact ground state

    Args:
        molecule: name of molecule
        collection_list: list of all collections

    Returns:
        sum of standard deviations, list of variances
    '''
    data = get_molecular_data()
    hamil, _, num_q, _, _, _ = data[molecule]
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
def hadamard_test_residual_variance(molecule: str, collection_list: list[list[QubitOperator]], ref_idx: int = 0) -> tuple[float, float, np.ndarray]:
    '''Measures residual variance for each term in Hamiltonian decomposition using Hadamard test circuit and exact ground state
    
    Args:
        molecule: name of molecule
        collection_list: list of all collections
        ref_idx: index in dets of ansatz (reference), default = 0

    Returns:
        (sum of standard deviations, sum of variances)
    '''
    data = get_molecular_data()
    hamil, flipped_dets, num_q, _, _, _ = data[molecule]
    dim = 2**num_q
    ground_state_vec = ground_state(hamil, num_q)[0]
    ground_state_vec = ground_state_vec.flatten()
    coll_hamils = collection_hamiltonians(collection_list, num_q)
    del flipped_dets[ref_idx]
    dets = [
        powers_of_2(
            flip_binary_bitstring(
                sum([2**k for k in flipped_det]),
                num_q
            )
        )
        for flipped_det in flipped_dets
    ]
    det_vecs = dets_to_statevector(dets, num_q)

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

    return (np.sum(summed_stdevs), np.sum(summed_vars), var_list)
