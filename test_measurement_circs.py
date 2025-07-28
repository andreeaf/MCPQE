import numpy as np
import galois
from openfermion import QubitOperator

# --- Measurement circuit generation for a collection of commuting Paulis ---

# 1. (Sorted Insertion paper method)
def operator_binary_mapping(num_q: int, pauli_strings: list[QubitOperator]) -> tuple[np.ndarray, list[float]]:
    '''Maps list of Pauli strings to binary matrix representation

    Args:
        num_q: number of qubits
        pauli_strings: list of Pauli strings
    
    Returns:
        (binary matrix, coefficients of Pauli strings)
    '''

    bin_matrix = np.zeros(shape=(2*num_q, len(pauli_strings)))
    coeffs = []
    
    for idx, pauli_string in enumerate(pauli_strings):
        temp = list(pauli_string.terms.keys())
        if () not in temp:
            (*operator_list,) = temp[0]
            for act_qubit, operator in operator_list:
                match operator:
                    case 'Z':
                        bin_matrix[act_qubit, idx] = 1
                    case 'X':
                        bin_matrix[act_qubit+num_q, idx] = 1
                    case 'Y':
                        bin_matrix[act_qubit, idx] = 1
                        bin_matrix[act_qubit+num_q, idx] = 1
        coeffs.append(list(pauli_string.terms.values())[0])

    return (bin_matrix, coeffs)


def binary_gauss_elim(bin_matrix: np.ndarray) -> np.ndarray:
    '''Gaussian elimination on binary matrix
    
    Args:
        bin_matrix: input binary matrix

    Returns:
        Gaussian eliminated binary matrix (row echelon form)
    '''

    mat = np.array(bin_matrix).copy().astype(int)  # input could be a Galois wrapped ndarray, need to convert back to numpy
    N, K = mat.shape
    row_idx = 0
    for col in range(K):
        pivot_rows = np.where(mat[row_idx:, col] == 1)[0]
        if pivot_rows.size == 0:
            continue

        pivot_row = pivot_rows[0] + row_idx
        if pivot_row != row_idx:
            mat[[row_idx, pivot_row]] = mat[[pivot_row, row_idx]]
        for row in range(N):
            if row != row_idx and mat[row, col] == 1:
                mat[row] ^= mat[row_idx]
        row_idx += 1
        if row_idx == N:
            break
    return mat


def extract_pivot_cols(bin_matrix: np.ndarray, gauss_elim_matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    '''Extracts a linearly independent set of columns of a binary matrix by finding pivot columns (with only one 1) in Gaussian eliminated matrix
    
    Args:
        bin_matrix: input binary matrix of size (N, K)
        gauss_elim_matrix: Gaussian eliminated matrix

    Returns:
        linearly independent columns of bin_matrix in the original order, indices of pivot columns
    '''

    _, K = bin_matrix.shape
    pivot_cols = []
    one_rows = []  # indices of rows containing 1 for columns with one 1
    for col in range(K):
        check_column = gauss_elim_matrix[:, col]
        nonzero_idxs = np.nonzero(check_column)[0]
        if len(nonzero_idxs) == 1 and nonzero_idxs[0] not in one_rows:  # disregard duplicate pivot columns
            pivot_cols.append(col)
            one_rows.append(nonzero_idxs[0])
    pivot_only_mat = bin_matrix[:, pivot_cols]

    return (pivot_only_mat, pivot_cols)


def rank(bin_matrix: np.ndarray) -> int:
    '''Rank of a binary matrix
    
    Args:
        bin_matrix: input binary matrix 
    
    Returns:
        rank: rank of bin_matrix
    '''

    copy_mat = np.array(bin_matrix).copy().astype(int)
    mat = binary_gauss_elim(copy_mat)

    rank = np.count_nonzero(np.any(mat, axis=1))

    return rank


def cZ_measurement(num_q: int, pauli_strings: list[QubitOperator], verbose=False) -> tuple[np.ndarray, dict[str, list[int | tuple[int, int]]]]:
    '''Full cZ measurement circuit
    
    Args:
        num_q: number of qubits
        pauli_strings: list of commuting collection of Pauli strings
        verbose: if True prints binary matrix at each stage

    Returns:
        (R matrix, measurement circuit)
    '''
    gates_to_apply = []
    GF = galois.GF(2)  # easy usage of numpy for binary matrices

    bin_matrix, coeffs = operator_binary_mapping(num_q, pauli_strings)
    if verbose:
        print("S'_start")
        print(bin_matrix, "\n")
    first_gauss_elim = binary_gauss_elim(bin_matrix)
    lin_indep_cols = extract_pivot_cols(bin_matrix, first_gauss_elim)[0]  # take only linearly independent Pauli strings
    if verbose:
        print("S_start")
        print(lin_indep_cols, "\n")
    N, start_rank = lin_indep_cols.shape
    n = N // 2

    # Matrix corresponding to first Gaussian elimination
    R0 = first_gauss_elim[0:start_rank].copy().astype(int)  # (this is a conjecture)
    if verbose:
        print("R_0")
        print(R0, "\n")

    # Rearrange on X submatrix so linearly independent rows are on top
    z_submat = lin_indep_cols[0:n]
    x_submat = lin_indep_cols[n:]
    rows_to_rearr = np.array(extract_pivot_cols(x_submat.T, binary_gauss_elim(x_submat.T))[1], dtype=int)  # rows to swap to the top of X and Z submatrices (qubit swap)
    num_rearr_rows = len(rows_to_rearr)
    all_rows = np.arange(n)
    rows_not_rearr = np.setdiff1d(all_rows, rows_to_rearr).astype(int)
    z_rearr = GF(z_submat[np.concatenate((rows_to_rearr, rows_not_rearr))].astype(int))
    x_rearr = GF(x_submat[np.concatenate((rows_to_rearr, rows_not_rearr))].astype(int))
    
    # Apply Hadamards to non-pivot rows (swap rows of X and Z submatrices)
    z_sub_h = z_rearr.copy()[num_rearr_rows:]
    x_sub_h = x_rearr.copy()[num_rearr_rows:]
    z_rearr[num_rearr_rows:] = x_sub_h
    x_rearr[num_rearr_rows:] = z_sub_h
    if verbose:
        print("S_1")
        print(z_rearr)
        print(x_rearr, "\n")
    gates_to_apply.append({'H': rows_not_rearr})

    # Refind and reswap pivot rows to get fullrank upper submatrix
    rows_rearr_post_h = np.array(extract_pivot_cols(x_rearr.T, binary_gauss_elim(x_rearr.T))[1], dtype=int)
    rows_not_rearr_post_h = np.setdiff1d(all_rows, rows_rearr_post_h).astype(int)
    z_re_rearr = GF(np.array(z_rearr[np.concatenate((rows_rearr_post_h, rows_not_rearr_post_h))]).astype(int))
    x_re_rearr = GF(np.array(x_rearr[np.concatenate((rows_rearr_post_h, rows_not_rearr_post_h))]).astype(int))
    if verbose:
        print("S_1 after post-Hadamard rearrangement")
        print(z_re_rearr)
        print(x_re_rearr, "\n")

    # Invert upper (k,k) submatrix of X submatrix, multiply by current
    x_upper_submat = x_re_rearr[0:start_rank]
    R1 = np.linalg.inv(x_upper_submat)  # x_upper_submat should be fullrank
    z_new_submat = z_re_rearr @ R1
    x_new_submat = x_re_rearr @ R1
    if verbose:
        print("R_1")
        print(R1, "\n")
        print("S_2")
        print(z_new_submat)
        print(x_new_submat, "\n")

    # Append columns to complete the basis
    x_top = np.hstack((x_new_submat.copy()[0:start_rank], np.zeros((start_rank, n - start_rank))))
    x_bottom = np.hstack((x_new_submat.copy()[start_rank:], np.eye(n - start_rank)))
    x_res = np.vstack((x_top, x_bottom))
    
    D = z_new_submat.copy()[start_rank:]
    z_top = np.hstack((z_new_submat.copy()[0:start_rank], D.T))
    z_bottom = np.hstack((D, np.zeros((n - start_rank, n - start_rank))))
    z_res = np.vstack((z_top, z_bottom))
    if verbose:
        print("S_3")
        print(z_res)
        print(x_res, "\n")
    R2 = GF(np.vstack((np.eye(start_rank), np.zeros((n - start_rank, start_rank)))).astype(int))
    if verbose:
        print("R_2")
        print(R2, "\n")
    
    # Processing to get S_4 = S_graph
    xres_temp = np.array(x_res, dtype=int)
    R3 = np.linalg.inv(GF(xres_temp))  # X submatrix should be (square and) fullrank
    S3 = np.vstack((z_res, x_res)).astype(np.uint8)
    S3R3 = S3 @ R3
    if verbose:
        print("S_3 * R_3")
        print(S3R3, "\n")

    phase_rows = []  # qubits on which to apply Phase gate
    for row in range(n):
        if S3R3[row, row] == 1:
            phase_rows.append(row)
            copy_S3R3 = S3R3.copy()
            new_row = copy_S3R3[row] ^ copy_S3R3[row + n]
            S3R3[row] = new_row
    gates_to_apply.append({'S': phase_rows})

    R = R3 @ R2 @ R1 @ GF(R0)
    if verbose:
        print("S_4")
        print(S3R3, "\n")
        print("R")
        print(R, "\n")
        
    cZ_gates = []  # qubit pairs on which to apply cZ
    for graph_row in range(1,n):
        for graph_col in range(graph_row):
            if S3R3[graph_row, graph_col] == 1:
                cZ_gates.append((graph_row, graph_col))
    gates_to_apply.append({'cZ': cZ_gates})
    gates_to_apply.append({'H': np.arange(n).tolist()})
    if verbose:
        print("GATES TO APPLY")
        print(gates_to_apply, "\n")

    return (np.array(R), gates_to_apply)