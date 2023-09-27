import numpy as np
from pytket.extensions.qiskit import AerBackend, AerStateBackend

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
            vector = mat.dot(mcirc.get_statevector())
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
    vector_square = np.conj(vector) * vector
    distribution = np.zeros(len(vector))
    probability = np.zeros(len(vector))
    prob = 0
    for i in range(len(vector_square)):
        if vector_square[i] > 1e-9:
            probability[i] = vector_square[i]
            prob += vector_square[i]
            distribution[i] = prob
            last_dist = i
    distribution[:]/=distribution[last_dist]
    if nshots == 0 or gaussian:
        counts = {mapping[a]:probability[a] for a in mapping}
        return counts
    counts = {}
    rands = np.random.rand(nshots)
    length = len(distribution)
    rang = range(length)
    for rand in rands:
        for j in rang:
            if rand < distribution[j]:
                if mapping[j] in counts:
                    counts[mapping[j]] +=1
                else:
                    counts[mapping[j]] =1
                break
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
