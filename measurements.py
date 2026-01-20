import bisect
import numpy as np
import utils
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from itertools import product
from hamil_decomp_utils import tensorproduct
from openfermion.linalg import qubit_operator_sparse

def generate_hamil_overlap_measure_circuits(mcirc, ps_g, nqubits):
    backend = AerBackend()
    circs = {}
    for pauli_gr in ps_g:
        measurement_circ = mcirc.copy()
        for index in range(len(pauli_gr[0])):
            if pauli_gr[0][index] == Pauli.X:
                measurement_circ.H(pauli_gr[1][index])
            if pauli_gr[0][index] == Pauli.Y:
                measurement_circ.V(pauli_gr[1][index])
        for index in range(nqubits):
            measurement_circ.add_bit(Bit(index+1))
        measurement_circ.measure_all()
        comp_circ = backend.get_compiled_circuit(measurement_circ)
        circs[pauli_gr] = comp_circ.copy()
    return circs

def generate_hamil_apply_circ(mcirc, ps_g, nqubits):
    backend = AerBackend()
    circs = {}
    for pauli_gr in ps_g:
        measurement_circ = mcirc.copy()
        for index in range(len(pauli_gr[0])):
            if pauli_gr[0][index] == Pauli.X:
                measurement_circ.X(pauli_gr[1][index]-1)
            if pauli_gr[0][index] == Pauli.Y:
                measurement_circ.Y(pauli_gr[1][index]-1)
            if pauli_gr[0][index] == Pauli.Z:
                measurement_circ.Z(pauli_gr[1][index]-1)
        for index in range(nqubits):
            measurement_circ.add_bit(Bit(index))
        measurement_circ.measure_all()
        comp_circ = backend.get_compiled_circuit(measurement_circ)
        circs[pauli_gr] = comp_circ.copy()
    return circs

def measure_hamil_overlap_shots(ps_g, circuits, n_shots):
    backend = AerBackend()
    hamil_elem = 0
    all_counts = []
    for pauli_group, circ in circuits.items():
        if pauli_group in ps_g:
            handle = backend.process_circuit(circ, n_shots = n_shots)
            counts = backend.get_result(handle).get_counts()
            for pauli_str in ps_g[pauli_group]:
                avg = 0
                for count in counts:
                    sign = 1
                    for index in pauli_str[2]:
                        if count[index] == 1:
                            sign *= -1
                    avg += sign * counts[count]
                hamil_elem += pauli_str[1] * avg/n_shots
            for count in counts:
                if count not in all_counts:
                    all_counts.append(count)
    return hamil_elem, all_counts

def measure_hamil_overlap_groups_matrices(ps_g, mats, mcirc, mapping, nshots = 0, gaussian = False):
    hamil_elem = 0
    for pauli_group, mat in mats.items():
        if pauli_group in ps_g:
            vector = mcirc.get_statevector()
            vector = mat.dot(vector)
            counts = get_counts(vector, nshots, mapping, gaussian)
            for pauli_str in ps_g[pauli_group]:
                avg = 0
                for count in counts:
                    sign = 1
                    for index in pauli_str[2]:
                        if count[index] == 1:
                            sign *= -1
                    avg += sign * counts[count]
                if nshots and not gaussian:
                    hamil_elem += pauli_str[1] * avg/nshots
                elif nshots and gaussian:
                    epsilon = 1/nshots
                    hamil_elem += pauli_str[1]*(avg + np.random.normal(0, epsilon))
                else:
                    hamil_elem += pauli_str[1]*avg
    return hamil_elem


def get_resvars(num_q, sq_mats_with_coeffs, mats_with_coeffs, mcirc, mapping, nshots=0, gaussian=False, shot_split=None):
    curr_statevector = mcirc.get_statevector()
    resvars = []

    for idx, coll_msmt_inputs in enumerate(mats_with_coeffs):
        hamil_overlap = 0
        msmt_mat, qubits_to_measure_list, phases, coll = coll_msmt_inputs
        sq_msmt_mat, sq_qubits_to_measure_list, sq_phases, sq_coeffs, iden_term = sq_mats_with_coeffs[idx]
        sq_hamil_overlap = iden_term
        statevec = msmt_mat.dot(curr_statevector)
        sq_statevec = sq_msmt_mat.dot(curr_statevector)
        if shot_split:
            nshots = shot_split[idx]
        counts = get_counts(statevec, nshots, mapping, gaussian)
        sq_counts = get_counts(sq_statevec, nshots, mapping, gaussian)

        for pauli_string, qubits_to_measure, qubit_parity_phase in zip(coll, qubits_to_measure_list, phases):
            coeff = list(pauli_string.terms.values())[0]
            for count in counts.keys():
                xor_val = 0
                for measured_qubit_idx in qubits_to_measure:
                    xor_val = xor_val ^ count[measured_qubit_idx]
                xor_phase = (-1) ** xor_val
                if nshots != 0:
                    hamil_overlap += coeff * xor_phase * qubit_parity_phase * counts[count]/nshots
                else:
                    hamil_overlap += coeff * xor_phase * qubit_parity_phase * counts[count]

        for sq_qubits_to_measure, sq_qubit_parity_phase, sq_coeff in zip(sq_qubits_to_measure_list, sq_phases, sq_coeffs):
            for sq_count in sq_counts.keys():
                sq_xor_val = 0
                for sq_msd_qubit_idx in sq_qubits_to_measure:
                    sq_xor_val = sq_xor_val ^ sq_count[sq_msd_qubit_idx]
                sq_xor_phase = (-1) ** sq_xor_val
                if nshots != 0:
                    sq_hamil_overlap += sq_coeff * sq_xor_phase * sq_qubit_parity_phase * sq_counts[sq_count]/nshots
                else:
                    sq_hamil_overlap += sq_coeff * sq_xor_phase * sq_qubit_parity_phase * sq_counts[sq_count]

        resvars.append(sq_hamil_overlap - hamil_overlap**2)

    return resvars


def measure_collection_overlap(num_q, mats_with_coeffs, mcirc, mapping, nshots=0, gaussian=False, shot_split=None):
    '''Sorted insertion <phi_i|H|Psi> term'''
    hamil_overlap = 0
    curr_statevector = mcirc.get_statevector()

    for coll_idx, coll_msmt_inputs in enumerate(mats_with_coeffs):
        msmt_mat, qubits_to_measure_list, phases, coll = coll_msmt_inputs
        statevec = msmt_mat.dot(curr_statevector)
        if shot_split:
            nshots = shot_split[coll_idx]
        counts = get_counts(statevec, nshots, mapping, gaussian)

        for pauli_string, qubits_to_measure, qubit_parity_phase in list(zip(coll, qubits_to_measure_list, phases)):
            coeff = list(pauli_string.terms.values())[0]
            for count in list(counts.keys()):
                xor_val = 0
                for measured_qubit_idx in qubits_to_measure:
                    xor_val = xor_val ^ count[measured_qubit_idx]
                xor_phase = (-1) ** xor_val
                if nshots != 0:
                    hamil_overlap += coeff * xor_phase * qubit_parity_phase * counts[count]/nshots
                else:
                    hamil_overlap += coeff * xor_phase * qubit_parity_phase * counts[count]
                
    return hamil_overlap


def measure_overlap_shots_mat(mcirc, mapping, nshots = 0, gaussian = False):
    overlap = 0
    vector = mcirc.get_statevector()
    counts = get_counts(vector, nshots, mapping, gaussian)
    avg = 0
    for count in counts:
        sign = 1
        if count[0] == 1:
            sign *= -1
        avg += sign * counts[count]
    if nshots and not gaussian:
        overlap = avg/nshots
    elif nshots and gaussian:
        epsilon = 1/nshots
        overlap = avg + np.random.normal(0,epsilon)
    else:
        overlap = avg
    return overlap

def get_counts(vector, nshots, mapping, gaussian = False):
    vector_square = np.conj(vector)*vector
    indices = []
    distribution = []
    prob = 0
    for i in range(len(vector_square)):
        if vector_square[i] > 1e-9:
            prob += vector_square[i]
            indices.append(i)
            distribution.append(prob)
    distribution[:]/=distribution[-1]
    if nshots == 0 or gaussian:
        counts = {mapping[a]:vector_square[a].real for a in mapping}
        return counts
    counts = {}
    rands = np.random.rand(nshots)
    for rand in rands:
        j = bisect.bisect_left(distribution, rand)
        if mapping[indices[j]] in counts:
            counts[mapping[indices[j]]] +=1
        else:
            counts[mapping[indices[j]]] =1
    return counts

def measure_hamil_overlap(mcirc, ps_g, nqubits, shots= False, n_shots = 0):
    '''Measure <D2|HU|D1>'''

    hamil_elem = 0
    if shots:
        backend = AerBackend()
        for pauli_gr in ps_g:
            measurement_circ = mcirc.copy()
            for index in range(len(pauli_gr[0])):
                if pauli_gr[0][index] == Pauli.X:
                    measurement_circ.H(pauli_gr[1][index])
                if pauli_gr[0][index] == Pauli.Y:
                    measurement_circ.V(pauli_gr[1][index])
            for index in range(nqubits):
                measurement_circ.add_bit(Bit(index+1))
            measurement_circ.measure_all()
            comp_circ = backend.get_compiled_circuit(measurement_circ)
            handle=backend.process_circuit(comp_circ, n_shots = n_shots)
            counts = backend.get_result(handle).get_counts()
            for pauli_str in ps_g[pauli_gr]:
                avg = 0
                for count in counts:
                    sign = 1
                    for index in pauli_str[2]:
                        if count[index] == 1:
                            sign *= -1
                    avg += sign * counts[count]
                hamil_elem += pauli_str[1] * avg/n_shots
    else:
        backend = AerStateBackend()
        comp_circ = backend.get_compiled_circuit(mcirc)
        for pauli_gr in ps_g:
            for pauli_str in ps_g[pauli_gr]:
                hamil_elem += pauli_str[1]*backend.get_pauli_expectation_value(comp_circ,
                                                 QubitPauliString([Qubit(x) for x in pauli_str[2]],
                                                 pauli_str[0]))
    return hamil_elem
def generate_overlap_measurement(mcirc):
    backend = AerBackend()
    measurement_circ = mcirc.copy()
    measurement_circ.Measure(0,0)
    comp_circ = backend.get_compiled_circuit(measurement_circ)
    return comp_circ

def measure_overlap_shots(mcirc, n_shots):
    backend = AerBackend()
    handle=backend.process_circuit(mcirc, n_shots = n_shots)
    counts = backend.get_result(handle).get_counts()
    return (counts[(0,)] - counts[(1,)])/(counts[(0,)] + counts[(1,)])

def measure_overlap(mcirc, shots = False, n_shots = 0):

    if shots:
        backend = AerBackend()
        measurement_circ = mcirc.copy()
        measurement_circ.Measure(0,0)
        comp_circ = backend.get_compiled_circuit(measurement_circ)
        handle=backend.process_circuit(comp_circ, n_shots = n_shots)
        counts = backend.get_result(handle).get_counts()
        return (counts[(0,)] - counts[(1,)])/(counts[(0,)] + counts[(1,)])

    sim = AerStateBackend()
    comp_circ = sim.get_compiled_circuit(mcirc)
    real_overlap = sim.get_pauli_expectation_value(comp_circ,
                                                   QubitPauliString(Qubit(0), Pauli.Z))
    return real_overlap
