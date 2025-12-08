import numpy as np
import galois
import pytket
from openfermion import QubitOperator
from openfermion.linalg import qubit_operator_sparse
from hamil_decomps import qubitwise_comm, sorted_insertion
from hamil_decomp_utils import powers_of_2, tensorproduct

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
        y_count = 0
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
                        y_count += 1
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


def add_ancilla_Z(op: QubitOperator) -> QubitOperator:
    '''Adds an ancilliary qubit at position 0 and applies Z'''

    new_term_str = "Z0"
    for pauli_str in op.terms.keys():
        for qubit_idx, gate in pauli_str:
            new_term_str += f" {gate}{qubit_idx+1}"

    return QubitOperator(new_term_str, list(op.terms.values())[0])


def swap_pairs_from_top_bottom(top_qubits: list[int], bottom_qubits: list[int]) -> list[tuple[int]]:
    '''Returns (the minimal number of) pairs of qubits to swap to produce permutation of the form [qubits at top, qubits at bottom]. 
    top_qubits and bottom_qubits must contain all qubit indices.

    Args:
        top_qubits: list of qubit indices at the top of the new permutation
        bottom_qubits: list of qubit indices below top_qubits in the new permutation

    Returns:
        swap_qubit_pairs: list of qubit index tuples to perform SWAP on, in order
    '''
    new_perm = top_qubits
    new_perm.extend(bottom_qubits)
    num_qubits = len(new_perm)
    visited = [False]*num_qubits
    swap_qubit_pairs = []

    for i in range(num_qubits):
        if visited[i] or new_perm[i] == i:
            continue

        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(j)
            j = new_perm[j]

        if len(cycle) > 1:
            qubit_1 = cycle[0]
            for cycled_qubit in reversed(cycle[1:]):
                swap_qubit_pairs.append((qubit_1, cycled_qubit))

    return swap_qubit_pairs


def swap_rows_diagonalise(unswapped_mat: np.ndarray) -> np.ndarray:
    '''Swaps rows of matrix until it is diagonal. Assumes this can be done.
    
    Args:
        unswapped_mat: matrix whose rows can be swapped to make it diagonal
    
    Returns:
        swapped_mat: row-swapped diagonal matrix
    '''
    dim = len(unswapped_mat)
    swapped_mat = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(dim):
            if unswapped_mat[j][i] != 0:
                swapped_mat[i] = unswapped_mat[j]
    
    return swapped_mat


def get_conj_table() -> dict:
    # dict of M: P: (phase, |M_dag * P * M|)

    conj_table = {
        'H': {
            'I': (1, 'I'),
            'X': (1, 'X'),
            'Y': (-1, 'Y'),
            'Z': (1, 'Z')
        }, 
        'CZ': {
            ('I', 'I'): (1, ('I', 'I')),
            ('I', 'X'): (1, ('Z', 'X')),
            ('I', 'Y'): (1, ('Z', 'Y')),
            ('I', 'Z'): (1, ('Z', 'I')),
            ('X', 'I'): (1, ('X', 'Z')),
            ('X', 'X'): (1, ('Y', 'Y')),
            ('X', 'Y'): (-1, ('Y', 'X')),
            ('X', 'Z'): (1, ('X', 'I')), 
            ('Y', 'I'): (1, ('Y', 'Z')),
            ('Y', 'X'): (-1, ('X', 'Y')),
            ('Y', 'Y'): (1, ('X', 'X')), 
            ('Y', 'Z'): (1, ('Y', 'I')), 
            ('Z', 'I'): (1, ('I', 'Z')), 
            ('Z', 'X'): (1, ('I', 'X')), 
            ('Z', 'Y'): (1, ('I', 'Y')),
            ('Z', 'Z'): (1, ('Z', 'Z'))
        }, 
        'S': {
            'I': (1, 'I'),
            'X': (-1, 'Y'),
            'Y': (-1, 'X'),
            'Z': (1, 'Z')
        }
    }

    return conj_table


def CZ_measurement(num_q: int, pauli_strings: list[QubitOperator], add_ancilla=True, verbose=False) -> tuple[np.ndarray,
                                                                                           pytket.Circuit,
                                                                                           list[float],
                                                                                           list[dict[str, list[int | tuple[int, int]]]]]:
    '''Full CZ measurement circuit
    
    Args:
        num_q: number of qubits
        pauli_strings: list of commuting collection of Pauli strings
        add_ancilla: add Z gate on ancilliary qubit
        verbose: if True prints binary matrix at each stage

    Returns:
        (R matrix, measurement circuit (pytket), phases, list of gates in measurement circuit)
    '''
    gates_to_apply = []
    GF = galois.GF(2)  # easy usage of numpy for binary matrices

    if add_ancilla:
        pauli_strings = [add_ancilla_Z(op) for op in pauli_strings]
        num_q += 1

    bin_matrix, _ = operator_binary_mapping(num_q, pauli_strings)
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
    R0_inv = first_gauss_elim[0:start_rank].copy().astype(int)  # (this is a conjecture)
    if verbose:
        print("R_0_inv")
        print(R0_inv, "\n")

    # Rearrange on X submatrix so linearly independent rows are on top
    z_submat = lin_indep_cols[0:n]
    x_submat = lin_indep_cols[n:]
    rows_to_rearr = np.array(extract_pivot_cols(x_submat.T, binary_gauss_elim(x_submat.T))[1], dtype=int)  # rows to swap to the top of X and Z submatrices (qubit swap)
    num_rearr_rows = len(rows_to_rearr)
    all_rows = np.arange(n)
    rows_not_rearr = np.setdiff1d(all_rows, rows_to_rearr).astype(int)
    new_ordering_1 = np.concatenate((rows_to_rearr, rows_not_rearr))
    z_rearr = GF(z_submat[new_ordering_1].astype(int))
    x_rearr = GF(x_submat[new_ordering_1].astype(int))
    
    # Apply Hadamards to non-pivot rows (swap rows of X and Z submatrices)
    z_sub_h = z_rearr.copy()[num_rearr_rows:]
    x_sub_h = x_rearr.copy()[num_rearr_rows:]
    z_rearr[num_rearr_rows:] = x_sub_h
    x_rearr[num_rearr_rows:] = z_sub_h
    if verbose:
        print("S_1")
        print(z_rearr)
        print(x_rearr, "\n")
    gates_to_apply.append({'H': rows_not_rearr})  # rows_not_rearr corresponds to original qubit indexing

    # Refind and reswap pivot rows to get fullrank upper submatrix
    rows_rearr_post_h = np.array(extract_pivot_cols(x_rearr.T, binary_gauss_elim(x_rearr.T))[1], dtype=int)
    rows_not_rearr_post_h = np.setdiff1d(all_rows, rows_rearr_post_h).astype(int)  # corresponds to qubit indexing after first rearrangement
    new_ordering_2 = np.concatenate((rows_rearr_post_h, rows_not_rearr_post_h))
    z_re_rearr = GF(np.array(z_rearr[new_ordering_2]).astype(int))
    x_re_rearr = GF(np.array(x_rearr[new_ordering_2]).astype(int))
    qubit_idx_mapping = [new_ordering_1[idx] for idx in new_ordering_2]
    if verbose:
        print("S_1 after post-Hadamard rearrangement")
        print(z_re_rearr)
        print(x_re_rearr, "\n")

    # Invert upper (k,k) submatrix of X submatrix, multiply by current
    x_upper_submat = x_re_rearr[0:start_rank]
    R1_inv = x_upper_submat
    R1 = np.linalg.inv(R1_inv)  # x_upper_submat should be fullrank
    z_new_submat = z_re_rearr @ R1
    x_new_submat = x_re_rearr @ R1
    if verbose:
        print("R_1_inv")
        print(x_upper_submat, "\n")
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
    R2_inv = GF(np.vstack((np.eye(start_rank), np.zeros((n - start_rank, start_rank)))).astype(int))
    if verbose:
        print("R_2_inv")
        print(R2_inv, "\n")
    
    # Processing to get S_4 = S_graph
    xres_temp = np.array(x_res, dtype=int)
    R3_inv = GF(xres_temp)
    R3 = np.linalg.inv(R3_inv)  # X submatrix should be (square and) fullrank
    S3 = np.vstack((z_res, x_res)).astype(np.uint8)
    S3R3 = S3 @ R3
    if verbose:
        print("R_3_inv")
        print(xres_temp, "\n")
        print("R_3")
        print(R3, "\n")
        print("S_3 * R_3")
        print(S3R3, "\n")

    phase_rows = []  # qubits on which to apply Phase gate, corresponds to qubit indexing after both arrangements
    for row in range(n):
        if S3R3[row, row] == 1:
            phase_rows.append(qubit_idx_mapping[row])
            copy_S3R3 = S3R3.copy()
            new_row = copy_S3R3[row] ^ copy_S3R3[row + n]
            S3R3[row] = new_row
    gates_to_apply.append({'S': phase_rows})

    R_inv = R3_inv @ R2_inv @ R1_inv @ GF(R0_inv)
    if verbose:
        print("S_4")
        print(S3R3, "\n")
        print("R_inv")
        print(R_inv, "\n")
        
    CZ_gates = []  # qubit pairs on which to apply cZ
    for graph_row in range(1,n):
        for graph_col in range(graph_row):
            if S3R3[graph_row, graph_col] == 1:
                CZ_gates.append((qubit_idx_mapping[graph_row], qubit_idx_mapping[graph_col]))
    gates_to_apply.append({'CZ': CZ_gates})
    gates_to_apply.append({'H': np.arange(n).tolist()})
    if verbose:
        print("GATES TO APPLY")
        print(gates_to_apply, "\n")

    msmt_circ = pytket.Circuit(num_q)

    for gate in gates_to_apply:
        for gate_name, qubits in gate.items():
            if gate_name == 'H':
                for qubit in qubits:
                    msmt_circ.H(qubit)
            elif gate_name == 'S':
                for qubit in qubits:
                    msmt_circ.S(qubit)
            elif gate_name == 'CZ':
                for qubit_1, qubit_2 in qubits:
                    msmt_circ.CZ(qubit_1, qubit_2)

    R_inv_rearr = [[] for _ in range(num_q)]
    for rearr_idx, actual_idx in enumerate(qubit_idx_mapping):
        R_inv_rearr[actual_idx] = R_inv[rearr_idx]

    R_inv_transpose = np.array(R_inv_rearr).T
    msmt_mat = msmt_circ.get_unitary()

    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    phases = []
    for idx, pstring in enumerate(pauli_strings):
        pauli_mat = qubit_operator_sparse(pstring, num_q).toarray()
        norm_pauli_mat = pauli_mat / list(pstring.terms.values())[0]
        Zs_mat = tensorproduct(*tuple(map(lambda x: Z if x == 1 else I, R_inv_transpose[idx])))
        conj_pauli_mat = np.round(msmt_mat @ norm_pauli_mat @ (msmt_mat.conj().T), 3).real
        phase = np.round(conj_pauli_mat[0][0] / Zs_mat[0][0]).real
        if not np.all(phase*conj_pauli_mat == Zs_mat):
            print("ERROR IN PHASE OF PAULI STRING MEASUREMENT!")
        phases.append(phase)
    
    return (R_inv_transpose, msmt_circ, phases, gates_to_apply)


# TESTING FOR USING MSMT CIRCUIT IN HADAMARD TEST
# ---  USING H2, CHANGE BELOW IF USING DIFFERENT MOLECULE ---
NUM_Q = 4  # excluding ancilla 
COLLS = sorted_insertion("h2")  # excluding ancilla
# ----------------------------------------------------------

# -------------------- FIXED CONSTANTS ---------------------
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)
DIM = 2**NUM_Q
ALL_DETS_BIN = [
    [1 if i in powers_of_2(j) else 0 for i in range(NUM_Q+1)]
    for j in range(2*DIM)
]  # including ancilla
# ----------------------------------------------------------

def get_cz_msmt_data():
    return [CZ_measurement(NUM_Q, coll) for coll in COLLS]

def determine_Zs(diag_elems: np.array, num_q: int):
    assert len(diag_elems) == 2**num_q
    Z_qubs = []
    for qub_idx in range(num_q):
        if diag_elems[2**(num_q-1-qub_idx)]*diag_elems[0] == -1:
            Z_qubs.append(qub_idx)
    
    return Z_qubs

def check_can_be_row_swapped_to_Zs(matrix: np.ndarray) -> bool:
    M, N = matrix.shape
    assert M == N

    is_valid = True
    for i in range(N):
        one_count = 0
        zero_count = 0
        for j in range(N):
            val = np.abs(matrix[i][j])
            if val == 1:
                one_count += 1
            elif val == 0:
                zero_count += 1
        if one_count != 1 or zero_count != (N-1):
            is_valid = False
            break
    
    return is_valid


# Example: use tensorproduct(Z,X,X,Y,Y) to get ndarray matrix for Pauli string ZXXYY
def test_conjugate_pauli_with_msmt(msmt_circuit: pytket.Circuit, pstring: QubitOperator | np.ndarray, add_ancilla: bool, R_matrix_row: np.ndarray) -> tuple[np.ndarray]:
    # if supplying pstring as QubitOperator (from COLLS, maybe), it will be normalised
    # msmt_circuit and R_matrix_row can be obtained from get_cz_msmt_data()

    if type(pstring) == QubitOperator:
        coeff = list(pstring.terms.values())[0]
        pstring_mat = qubit_operator_sparse(pstring, NUM_Q).toarray() / coeff
    elif type(pstring) == np.ndarray:
        pstring_mat = pstring

    if add_ancilla:
        pstring_mat = np.kron(Z, pstring_mat)
    
    msmt_mat = msmt_circuit.get_unitary()
    conj_pauli = msmt_mat @ pstring_mat @ (msmt_mat.conj().T)

    if not np.all(np.isreal(conj_pauli)):
        print("Error: conjugated Pauli is not real")
        return conj_pauli, pstring_mat
    
    Zs_mat_list = []
    for qub in R_matrix_row:
        if qub == 0:
            Zs_mat_list.append(I)
        else:
            Zs_mat_list.append(Z)
    Zs_mat = tensorproduct(*Zs_mat_list)

    conj_pauli = np.round(conj_pauli.real, 2)  # round to 2 decimal places i.e. |tolerance| = 0.005 for diagonalisation
    if np.all(Zs_mat == conj_pauli):
        print("SUCCESS WITHOUT ROW SWAPS")
        return conj_pauli
    elif np.all(Zs_mat == (-1)*conj_pauli):
        print("SUCCESS with phase = -1")
        return conj_pauli, Zs_mat

    if not check_can_be_row_swapped_to_Zs(conj_pauli):
        print("Error: conjugated Pauli cannot be row-swapped into a valid matrix of Zs and Is")
        return conj_pauli, pstring_mat
    
    diag_conj_pauli = swap_rows_diagonalise(conj_pauli)  # diagonalised by row swaps (require computation of full matrices so implausible)
    if np.all(Zs_mat == diag_conj_pauli):
        print("SUCCESS with row swaps")
        return conj_pauli, diag_conj_pauli
    elif np.all(Zs_mat == (-1)*diag_conj_pauli):
        print("SUCCESS with row swaps and phase = -1")
        return conj_pauli, diag_conj_pauli, Zs_mat

    Zs_diag = np.diag(Zs_mat)
    swapped_conj_pauli_diag = np.diag(diag_conj_pauli)
    print("FAILURE")

    return conj_pauli, swapped_conj_pauli_diag, Zs_diag
    

def idx_from_binary(bin: list[int]):
    sum = 0
    for i, val in enumerate(bin):
        sum += val * (2**i)
    return sum


def check_coeff(coll_idx, pstring_idx, det_idx):
    CZ_MSMT_DATA = get_cz_msmt_data()
    COLL = COLLS[coll_idx]
    PSTRING = COLL[pstring_idx]
    PSTRING_COEFF = list(PSTRING.terms.values())[0]
    R, msmt_circ, _, _ = CZ_MSMT_DATA[coll_idx]
    msmt_circ_mat = msmt_circ.get_unitary()

    statevec = [2*np.random.rand()-1 for _ in range(DIM)]  # pick a random real-coefficient statevector, put outside the function to prevent repicking
    statevec = np.array(statevec) / np.linalg.norm(statevec)

    pstring_mat = qubit_operator_sparse(PSTRING, NUM_Q).toarray()
    correct_coeff = (pstring_mat @ statevec)[det_idx]

    msmt_det_statevec = np.zeros(DIM)
    msmt_det_statevec[det_idx] = 1
    h_test_statevec = (0.5) * np.concatenate(((msmt_det_statevec+statevec), (msmt_det_statevec-statevec)))
    h_test_pstring_mat = np.kron(Z, pstring_mat)
    h_test_overlap = (h_test_statevec.conj()) @ h_test_pstring_mat @ h_test_statevec

    qubits_to_measure = [i for i in range(NUM_Q+1) if R[pstring_idx][i] == 1]
    xor_vals = []
    for det_bin in ALL_DETS_BIN:
        xor_val = 0
        for qub_idx in qubits_to_measure:
            xor_val = xor_val ^ det_bin[qub_idx]
        xor_vals.append(xor_val)
    
    msmt_res_statevec = msmt_circ_mat @ h_test_statevec
    msmt_res_probs = np.multiply(msmt_res_statevec.conj(), msmt_res_statevec).real
    msmt_coeff = 0
    for idx, xor_val in enumerate(xor_vals):
        phase = (-1) ** (xor_val)
        msmt_coeff += phase * msmt_res_probs[idx] * PSTRING_COEFF
    return(correct_coeff, h_test_overlap, msmt_coeff)