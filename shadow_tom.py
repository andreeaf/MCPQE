import numpy as np
import time
import pickle as pkl

from itertools import permutations
from random import Random
from qiskit.quantum_info import random_clifford, Statevector
from hamil_decomp_utils import ground_state, dets_to_statevector, get_molecular_data, powers_of_2, flip_binary_bitstring
from openfermion import jordan_wigner, qubit_operator_sparse
from functools import reduce


def excitations(ref_det: tuple[int], curr_det: tuple[int], n: int = None) -> tuple[tuple[int], tuple[int], bool, bool | None]:
    '''Computes degree of excitation of determinant from reference determinant
    
    Args:
        ref_det (tuple[int]): reference determinant
        curr_det (tuple[int]): excited determinant
        n (int): check if curr_det is at most n excitations from ref_det

    Returns:
        annihilated orbitals,\n
        created orbitals,\n
        is excited determinant a double excitation,\n
        is excited determinant at most an n-excitation or None
    '''
    set_ref = set(ref_det)
    set_det = set(curr_det)

    annihilated_orbitals = tuple(set_ref - set_det)
    created_orbitals = tuple(set_det - set_ref)
    num_annihilated = len(annihilated_orbitals)
    num_created = len(created_orbitals)

    is_double_excit = bool(num_annihilated == num_created == 2)
    is_maximally_n_excit = bool(num_annihilated == num_created <= n) if n else None

    return annihilated_orbitals, created_orbitals, is_double_excit, is_maximally_n_excit


def median_of_means(samples_amplitudes, num_means: int, dim: int, det_list: list[tuple[int]]):
    '''Shuffles and splits amplitudes into equal chunks, returns the median of means of the chunks
    
    Args:
        samples_amplitudes (list[list[np.complex128]]): list of amplitudes for each shot
        num_means (int): number of chunks to split amplitudes data into
        dim (int): dimension of Hilbert space
        det_list (list[tuple[int]]): list of determinants in the same order as samples_amplitudes

    Returns:
        amplitudes_dict (dict[tuple[int], np.complex128]): dictionary of complex amplitude values by determinant
    '''
    NUM_MEANS = num_means
    DIM = dim
    rand_inst = Random()

    samples_amplitudes_split = samples_amplitudes.copy()
    rand_inst.shuffle(samples_amplitudes_split)
    samples_amplitudes_split = np.array_split(samples_amplitudes_split, NUM_MEANS)

    amplitude_means = []
    for amplitudes_split in samples_amplitudes_split:
        amplitude_means.append(2 * (DIM + 1) * np.sum(np.array(amplitudes_split), axis=0) / len(amplitudes_split))

    amplitudes_dict = {}
    det_wise_amplitude_means = np.array(amplitude_means).T
    for det_idx, det in enumerate(det_list):
        amplitudes_dict[det] = np.median(det_wise_amplitude_means[det_idx])

    return amplitudes_dict


def median_of_means_singles(samples_amplitudes, num_means: int, dim: int, det_list: list[tuple[int]]):
    NUM_MEANS = num_means
    DIM = dim
    rand_inst = Random()

    samples_amplitudes_split = samples_amplitudes.copy()
    rand_inst.shuffle(samples_amplitudes_split)
    samples_amplitudes_split = np.array_split(samples_amplitudes_split, NUM_MEANS)

    amplitude_means = [np.average(amplitudes_split, axis=0) for amplitudes_split in samples_amplitudes_split]
    amplitudes_dict = {}
    det_wise_amplitude_means = np.array(amplitude_means).T
    for det_idx, det in enumerate(det_list):
        amplitudes_dict[det] = np.median(det_wise_amplitude_means[det_idx])

    return amplitudes_dict


def tensorproduct(*args: np.ndarray) -> np.ndarray:
    return reduce(np.kron, args)


def classical_shadows(molecule: str, num_samples: int, num_measurements: int, method: int, num_means: int = 10, statevector: np.ndarray = None, calc_metrics: bool = False):
    '''Clifford shadows tomography on given state or ground state, with multiple measurements per Clifford and a median of means approach
    
    Args:
        molecule (str): name of molecule
        num_samples (int): number of random Cliffords to generate
        num_measurements (int): number of measurements per Clifford
        method (int): 
            0: N-qubit Clifford group
            1: 1-qubit Clifford group with integration over U(N)
            2: 1-qubit Clifford group with integration over U(1)
        num_means (int): number of means to divide all shots into (must divide num_samples * num_measurements, must be bigger than 1)
        statevector (np.ndarray): input quantum state (if None exact ground state is used), shape must be (2**n_qubits, 1)
        calc_metrics (bool): control calculation of performance metrics  (if True exact ground state is computed)

    Returns:
        amplitudes (coefficients) of double excitation determinants,\n
        amplitudes (coefficients) of non double excitation determinants,\n
        dictionary of energy residuals by determinant,\n
        (fidelity, calculated - actual ground state energy, (energies calculated with sign difference), ground state energy) or None
    '''
    data = get_molecular_data()
    hamil, allowed_dets, num_q, _, _, actual_hf_energy = data[molecule]
    NUM_SAMPLES = num_samples
    NUM_Q = num_q
    DIM = 2**NUM_Q  # dimension of Hilbert space
    DECIMAL_PRECISION = 7
    NUM_MEASUREMENTS = num_measurements
    NUM_MEANS = num_means
    VACUUM_IDX = 0  # index of vacuum state
    METHOD = method
    rand_inst = Random()
    
    # gates
    I = np.eye(2)
    H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]])
    F = (1/np.sqrt(2)) * np.array([[1,1j],[1,-1j]])
    CLIFFORD_SINGLE_GATES = (H,F,I)

    allowed_dets_flipped = [powers_of_2(flip_binary_bitstring(sum([2**k for k in det]), NUM_Q))
                            for det in allowed_dets
                            ]
    ref_det_flipped = allowed_dets[0]
    # rest of the function follows flipped determinant convention (e.g. HF determinant for H4 is (4,5,6,7))
    ref_det = powers_of_2(flip_binary_bitstring(sum([2**k for k in ref_det_flipped]), NUM_Q))  # Hartree-Fock determinant

    if statevector == None or calc_metrics:  # testing when exact ground state can be computed
        gs_vec, gs_energy = ground_state(hamil, NUM_Q)
        statevector = gs_vec.flatten()

    hamil_operator_list = list(hamil.get_operators())
    hamil_operator_dict = {}
    for op in hamil_operator_list:
        for key, val in op.terms.items():
            hamil_operator_dict[key] = val

    hf_idx = sum([2**k for k in ref_det])
    qubit_hamil = jordan_wigner(hamil)
    qubit_hamil.compress()
    qubit_hamil = qubit_operator_sparse(qubit_hamil, NUM_Q).real
    hf_energy = qubit_hamil[hf_idx, hf_idx]  # Hartree-Fock energy
    c0 = gs_vec[hf_idx][0]

    statevec_nonzero_idxs = np.nonzero(np.round(statevector, DECIMAL_PRECISION))[0]
    statevec_nonzero_dets = list(map(powers_of_2, statevec_nonzero_idxs))
    statevec_nonzero_basis_vecs = dets_to_statevector(statevec_nonzero_dets, NUM_Q)
    double_excite_dets = {}
    non_double_excite_dets = {}

    for det_idx, det in enumerate(statevec_nonzero_dets):
        annihilated_orbitals, created_orbitals, is_double_excit, _ = excitations(ref_det, det)
        det_statevec = statevec_nonzero_basis_vecs[det_idx]
        basis_idx = sum([2**k for k in det])
        if is_double_excit:
            det_op_key_perms = list(permutations(((created_orbitals[0], 1), (created_orbitals[1], 1), (annihilated_orbitals[0], 0), (annihilated_orbitals[1], 0))))
            for det_op_key in det_op_key_perms:
                if det_op_key in hamil_operator_dict.keys():
                    coeff = hamil_operator_dict[det_op_key]
                    annihilated_orbitals = (det_op_key[2][0], det_op_key[3][0])
                    created_orbitals = (det_op_key[0][0], det_op_key[1][0])
            double_excite_dets[det] = [(annihilated_orbitals, created_orbitals), coeff, det_statevec, basis_idx]
        else:
            non_double_excite_dets[det] = [(annihilated_orbitals, created_orbitals), det_statevec, basis_idx]

    vacuum_state = np.zeros(DIM)
    vacuum_state[VACUUM_IDX] = 1
    statevector = (statevector/np.sqrt(2)) + vacuum_state/np.sqrt(2)
    statevector = statevector / np.linalg.norm(statevector)

    double_excite_samples_amplitudes = []
    double_excite_dets_list = list(double_excite_dets.keys())

    non_double_excite_samples_amplitudes = []
    non_double_excite_dets_list = list(non_double_excite_dets.keys())
    
    if METHOD == 0:
        clifford_mats = [random_clifford(NUM_Q).to_matrix() for _ in range(NUM_SAMPLES)]

    if METHOD == 0 or METHOD == 1:
        for sample in range(NUM_SAMPLES):
            if sample != 0 and sample % (NUM_SAMPLES // 10) == 0:
                percent = (sample * 100) // NUM_SAMPLES
                dashes = 2 * percent // 10
                print(f"{'-' * dashes}{percent}%")
            if METHOD == 0:
                clifford = clifford_mats[sample]
            else:
                clifford = tensorproduct(*(CLIFFORD_SINGLE_GATES[i] for i in np.random.randint(0,3,NUM_Q)))
            res_state = Statevector(clifford @ statevector)
            res_state = res_state / np.linalg.norm(res_state.data)
            for _ in range(NUM_MEASUREMENTS):
                measured_bitstring = res_state.measure()[0][::-1]
                measured_idx = sum([int(j) * (2**i) for i, j in enumerate(measured_bitstring)])
                vac_proj_term = clifford[measured_idx][VACUUM_IDX]

                double_excite_sample_amplitudes = []
                for det_idx, det in enumerate(double_excite_dets_list):
                    _, _, _, det_statevec_idx = double_excite_dets[det]
                    c = np.conj(clifford[measured_idx][det_statevec_idx]) * vac_proj_term
                    double_excite_sample_amplitudes.append(c)
                double_excite_samples_amplitudes.append(double_excite_sample_amplitudes)

                non_double_excite_sample_amplitudes = []
                for det_idx, det in enumerate(non_double_excite_dets_list):
                    _, _, det_statevec_idx = non_double_excite_dets[det]
                    c = np.conj(clifford[measured_idx][det_statevec_idx]) * vac_proj_term
                    non_double_excite_sample_amplitudes.append(c)
                non_double_excite_samples_amplitudes.append(non_double_excite_sample_amplitudes) 

        double_excite_amplitudes = median_of_means(double_excite_samples_amplitudes, NUM_MEANS, DIM, double_excite_dets_list)
        non_double_excite_amplitudes = median_of_means(non_double_excite_samples_amplitudes, NUM_MEANS, DIM, non_double_excite_dets_list)

    elif METHOD == 2:
        for sample in range(NUM_SAMPLES):
            if sample != 0 and sample % (NUM_SAMPLES // 10) == 0:
                percent = (sample * 100) // NUM_SAMPLES
                dashes = 2 * percent // 10
                print(f"{'-' * dashes}{percent}%")
            single_qub_cliff_mats = [CLIFFORD_SINGLE_GATES[i] for i in np.random.randint(0,3,NUM_Q)]
            single_qub_cliff_mats_herm = [np.conjugate(mat.T) for mat in single_qub_cliff_mats]
            clifford = tensorproduct(*single_qub_cliff_mats)
            res_state = Statevector(clifford @ statevector)
            res_state = res_state / np.linalg.norm(res_state.data)
            for _ in range(NUM_MEASUREMENTS):
                measured_bitstring = res_state.measure()[0][::-1]
                density_mat_components = []
                for num, mat, matH in zip(measured_bitstring, single_qub_cliff_mats, single_qub_cliff_mats_herm):
                    ketbra_bitstring_mat = np.array([[1,0],[0,0]]) if num == "0" else np.array([[0,0],[0,1]])
                    density_mat_components.append(3 * matH @ ketbra_bitstring_mat @ mat - I)
                density_mat = tensorproduct(*density_mat_components)

                double_excite_coeffs = []
                for det in double_excite_dets_list:
                    _,_,_,det_idx = double_excite_dets[det]
                    double_excite_coeffs.append(density_mat[0][det_idx] + density_mat[det_idx][0])
                
                non_double_excite_coeffs = []
                for det in non_double_excite_dets_list:
                    _,_,det_idx = non_double_excite_dets[det]
                    non_double_excite_coeffs.append(density_mat[0][det_idx] + density_mat[det_idx][0])

                double_excite_samples_amplitudes.append(double_excite_coeffs)
                non_double_excite_samples_amplitudes.append(non_double_excite_coeffs)    

        double_excite_amplitudes = median_of_means_singles(double_excite_samples_amplitudes, NUM_MEANS, DIM, double_excite_dets_list)
        non_double_excite_amplitudes = median_of_means_singles(non_double_excite_samples_amplitudes, NUM_MEANS, DIM, non_double_excite_dets_list)

    calc_energy = 0
    for det, det_info in double_excite_dets.items():
        calc_energy += double_excite_amplitudes[det].real * qubit_hamil[hf_idx, det_info[3]] / c0
    energies = np.array([hf_energy + calc_energy, hf_energy - calc_energy])  # account for possible sign error
    correct_energy = energies[0] if np.sign(calc_energy) == np.sign(hf_energy) else energies[1]  

    hamil_minus_gs_energy = qubit_hamil - correct_energy * np.eye(DIM)
    resids = {}

    for det in allowed_dets_flipped:
        resid = 0
        det_statevec_idx = sum([2**k for k in det])
        for double_excite_det, double_excite_det_info in double_excite_dets.items():
            double_excite_det_idx = double_excite_det_info[3]
            resid += hamil_minus_gs_energy[det_statevec_idx, double_excite_det_idx] * double_excite_amplitudes[double_excite_det].real
        for non_double_excite_det, non_double_excite_det_info in non_double_excite_dets.items():
            non_double_excite_det_idx = non_double_excite_det_info[2]
            resid += hamil_minus_gs_energy[det_statevec_idx, non_double_excite_det_idx] * non_double_excite_amplitudes[non_double_excite_det].real
        resids[det] = resid

    metrics_tuple = None
    if calc_metrics:
        # Performance of coefficient measurement
        gs_double_excite_coeffs = {}
        for det in double_excite_dets_list:
            idx = sum([2**k for k in det])
            gs_double_excite_coeffs[det] = gs_vec[idx][0]
        gs_non_double_excite_coeffs = {}
        for det in non_double_excite_dets_list:
            idx = sum([2**k for k in det])
            gs_non_double_excite_coeffs[det] = gs_vec[idx][0]
        
        # 1. fidelity = 1 - <solution coefficients | actual coefficients>, normalised coefficient vectors
        solution_coeff_vector = []
        actual_coeff_vector = []
        for det in double_excite_dets_list:
            solution_coeff_vector.append(np.abs(double_excite_amplitudes[det].real))
            actual_coeff_vector.append(np.abs(gs_double_excite_coeffs[det]))
        solution_coeff_vector = np.array(solution_coeff_vector) / np.linalg.norm(solution_coeff_vector)
        actual_coeff_vector = np.array(actual_coeff_vector) / np.linalg.norm(actual_coeff_vector)
        fidelity = 1 - solution_coeff_vector @ actual_coeff_vector

        # 2. difference between calculated energy and ground state
        closest_energy_idx = np.argmin(np.abs(energies - gs_energy))
        correct_energy = energies[closest_energy_idx]
        min_energy_diff = energies[closest_energy_idx] - gs_energy

        # 3. fidelity including non-double excitations
        solution_coeff_vector = []
        actual_coeff_vector = []
        for det in double_excite_dets_list:
            solution_coeff_vector.append(np.abs(double_excite_amplitudes[det].real))
            actual_coeff_vector.append(np.abs(gs_double_excite_coeffs[det]))
        for det in non_double_excite_dets_list:
            solution_coeff_vector.append(np.abs(non_double_excite_amplitudes[det].real))
            actual_coeff_vector.append(np.abs(gs_non_double_excite_coeffs[det]))
        solution_coeff_vector = np.array(solution_coeff_vector) / np.linalg.norm(solution_coeff_vector)
        actual_coeff_vector = np.array(actual_coeff_vector) / np.linalg.norm(actual_coeff_vector)
        t_fidelity = 1 - solution_coeff_vector @ actual_coeff_vector
        
        metrics_tuple = (fidelity, min_energy_diff, energies, gs_energy, t_fidelity)

    return double_excite_amplitudes, non_double_excite_amplitudes, resids, metrics_tuple


def testing_classical_shadows(molecule: str, num_runs: int, method: int):
    NUM_RUNS = num_runs
    N_M_DICT = {
        # 50000: [(1000, 50), (1250, 40), (2000, 25), (2500, 20), (5000, 10)],
        # 80000: [(1000, 80), (1250, 64), (2000, 40), (2500, 32), (5000, 16), (8000, 10)],
        # 100000: [(1000, 100), (1250, 80), (2000, 50), (2500, 40), (5000, 20), (10000, 10)],
        # 150000: [(1000, 150), (1250, 120), (2000, 75), (2500, 60), (5000, 30), (10000, 15), (15000, 10)],
        200000: [(1000, 200), (2000, 100), (5000, 40), (10000, 20), (20000, 10)],
        500000: [(1000, 500), (2000, 250), (5000, 100), (10000, 50), (20000, 25), (50000, 10)],
        1000000: [(1000, 1000), (2000, 500), (5000, 200), (10000, 100), (50000, 20), (100000, 10)],
        # 2000000: [(5000, 400), (10000, 200), (50000, 40), (100000, 20), (200000, 10)],
        # 5000000: [(1000, 5000), (5000, 1000), (10000, 500), (50000, 100), (100000, 50), (200000, 25), (500000, 10)]
    }

    res_dict = {}
    for total_msmts, n_m_vals in N_M_DICT.items():
        print(f"TOTAL MEASUREMENTS: {total_msmts}")
        total_msmt_dict = {}
        for n,m in n_m_vals:
            print(f"N = {n}, M = {m}")
            res_arr = []
            for run in range(NUM_RUNS):
                print(f"RUN {run+1}")
                start_time = time.time()
                a,b,c,d = classical_shadows(molecule, n, m, method, calc_metrics=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"RUN DONE. ELAPSED TIME: {elapsed_time}")
                res_arr.append([d[0],d[1],elapsed_time,d[4],a,b,c])
                # [fidelity, energy diff, runtime, double excite coeffs, non double excite coeffs, residuals]
            total_msmt_dict[(n,m)] = res_arr
            filename = f"temp_{molecule}_shadows_{total_msmts}_method_{method}_{n}_{m}.pkl"
            with open(filename, "wb") as f:
                pkl.dump(res_arr, f)
            print("\n")
        print("\n")
        res_dict[total_msmts] = total_msmt_dict

    return res_dict

import matplotlib.pyplot as plt
def test_plot(molecule: str, num_runs: int, res_dict, metric_idx: int, errorbars: bool = True, include_title: bool = True):
    # metric = 0: fidelity, 1: energy diff, 2: runtime, 3: fidelity including non-double excitations
    RES_DICT = res_dict
    METRIC_IDX = metric_idx
    MOLECULE_STR = molecule.upper()
    NUM_RUNS = num_runs

    x_label = r"$\text{log}_{10} N$"
    if METRIC_IDX == 0:
        y_label = "fidelity = 1 - <computed coeffs|actual coeffs>"
        title = f"{MOLECULE_STR} fidelity, average over {NUM_RUNS} runs"
    elif METRIC_IDX == 1:
        y_label = "|computed - actual ground state energy| (hartree/molecule)"
        title = f"{MOLECULE_STR} energy difference, average over {NUM_RUNS} runs"
    elif METRIC_IDX == 2:
        y_label = "runtime (s)"
        title = f"{MOLECULE_STR} runtime, average over {NUM_RUNS} runs"
    elif METRIC_IDX == 3:
        y_label = "fidelity = 1 - <computed coeffs|actual coeffs>"
        title = f"{MOLECULE_STR} fidelity including non-double excitations, average over {NUM_RUNS} runs"
    else:
        y_label = ""
        title = ""

    plt.figure()
    for nm, nm_data in RES_DICT.items():
        line_label = rf"$NM = {nm}$"
        log_n_vals = []
        avg_data = []
        upper_err = [] if errorbars else None
        lower_err = [] if errorbars else None
        for n_m_tuple, data in nm_data.items():
            n, _ = n_m_tuple
            log_n_vals.append(np.log10(n))
            datum = [k[METRIC_IDX] for k in data]
            avg = np.average(datum)
            avg_data.append(avg)
            if errorbars:
                upper_err.append(np.max(datum) - avg)
                lower_err.append(avg - np.min(datum))
        plt.errorbar(log_n_vals, avg_data, (lower_err, upper_err), fmt='-o', capsize=5, label=line_label)
    
    if include_title:
        plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if METRIC_IDX == 1:
        plt.axhline(-0.00159, color='black')
        plt.axhline(0.00159, color='black')

    plt.show()


# script for running data collection remotely
# if __name__ == "__main__":
#     res_dict = testing_classical_shadows(molecule="h4", num_runs=10, method=0)
#     with open("shadows_h4_method0_2E5_5E5_1M.pkl", "wb") as f:
#         pkl.dump(res_dict, f)