import numpy as np
import copy

from hamil_decomp_utils import get_molecular_data, ground_state, collections_paulis_covariance_matrix
from hamil_decomps import add_forwards_compatible_paulis, collection_vars_ground_state, hadamard_test_residual_variance

def iterative_coefficient_splitting_naive(molecule: str, 
                                          num_meas: int, 
                                          num_iter: int) -> tuple[list[list[list[np.float64]]], list[list[int]]]:
    data = get_molecular_data()
    hamil, _, num_q, _, _, _ = data[molecule]
    
    
    collections_list, ops_locations, coeffs_matrix = add_forwards_compatible_paulis(molecule)
    num_collections = len(collections_list)
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

    gs_vec = ground_state(hamil, num_q)[0]
    colls_covariance_mats = collections_paulis_covariance_matrix(collections_list, num_q, gs_vec)  
    initial_collection_variances = []
    for i, coll in enumerate(collections_list):
        var = 0
        for j in range(len(coll)):
            for k in range(len(coll)):
                var += colls_covariance_mats[i][j,k]*coeffs_matrix[i][j]*coeffs_matrix[i][k]
        initial_collection_variances.append(var)
    M = num_meas
    varH = (sum(np.sqrt(x) for x in initial_collection_variances))**2/M
    measurement_nums = [np.ceil(np.sqrt(x) * sum(np.sqrt(y) for y in initial_collection_variances)/varH) for x in initial_collection_variances]
    all_msmt_lists = []  
    all_msmt_lists.append(measurement_nums)
    num_params = sum(len(x) for x in collections_list)
    ref_colls = {op: ops_locations[op][0] for op in ops_locations}
    # print(-1, varH, measurement_nums)
    for it in range(num_iter):
        matrix = np.zeros((num_params, num_params))
        result = np.zeros(num_params)
        meas = all_msmt_lists[-1]

        #Compute matrix and result vector for linear equations
        for index, coll in enumerate(collections_list):
            for j, op in enumerate(coll):
                ind1 = sum(len(x) for x in collections_list[:index]) + j
                op_key = list(op.terms.keys())[0]
                if index != ref_colls[op_key][0]:
                    for k, op2 in enumerate(coll):
                        ind2 = sum(len(x) for x in collections_list[:index]) + k
                        matrix[ind1, ind2] = colls_covariance_mats[index][j,k]/meas[index]        
                    for k, op2 in enumerate(collections_list[ref_colls[op_key][0]]):
                        ind2 = sum(len(x) for x in collections_list[:ref_colls[op_key][0]]) + k
                        matrix[ind1, ind2] = -colls_covariance_mats[ref_colls[op_key][0]][ref_colls[op_key][1],k]/meas[ref_colls[op_key][0]]        
                else:
                    for loc in ops_locations[op_key]:
                        matrix[ind1][sum(len(x) for x in collections_list[:loc[0]]) + loc[1]] = 1
                        result[ind1] = init_coeffs[op_key]

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
        collection_variances = []
        for i, coll in enumerate(collections_list):
            var = 0
            for j in range(len(coll)):
                for k in range(len(coll)):
                    var += colls_covariance_mats[i][j,k]*coeffs_matrix[i][j]*coeffs_matrix[i][k]
            collection_variances.append(var)
        # varH = (sum(np.sqrt(x) for x in collection_variances))**2/M
        # measurement_nums = [np.ceil(np.sqrt(x) * sum(np.sqrt(y) for y in collection_variances)/varH) for x in collection_variances]
        # all_msmt_lists.append(measurement_nums)
        measurement_nums = [np.ceil((M / np.sum([np.sqrt(y) for y in collection_variances])) * np.sqrt(x)) for x in collection_variances]
        all_msmt_lists.append(measurement_nums)

        # print(it, varH, measurement_nums)

    return all_coeffs_matrices, all_msmt_lists

# if __name__ == "__main__":
#     iterative_coefficient_splitting_naive('h4', 1000, 10)