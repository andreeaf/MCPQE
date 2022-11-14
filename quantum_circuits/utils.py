from pytket.pauli import Pauli, QubitPauliString
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.circuit import Circuit, Qubit, Bit
from openfermion.transforms import normal_ordered
from openfermion.ops import FermionOperator

import numpy as np

def pauli_strings_from_fermion_operators(operators, params, ancilla = 0):
    pauli_gate = {'X': Pauli.X, 'Y': Pauli.Y, 'Z': Pauli.Z, 'I': Pauli.I}
    pauli_strings = []
    k = 0
    for operator in operators:
        for term, coeff in operator.terms.items():
            paulis = []
            qubits = []
            for gate in term:
                paulis.append(pauli_gate[gate[1]])
                qubits.append(gate[0] + ancilla)
            pauli_strings.append([paulis, params[k]*coeff.imag/np.pi*2, qubits])
        k += 1
    return pauli_strings


def pauli_strings_from_fermion_operators_measurements(operators):
    pauli_gate = {'X': Pauli.X, 'Y': Pauli.Y, 'Z': Pauli.Z, 'I': Pauli.I}
    pauli_strings = []
    for operator in operators:
        for term, coeff in operator.terms.items():
            paulis = [Pauli.Z]
            qubits = [0]
            for gate in term:
                paulis.append(pauli_gate[gate[1]])
                qubits.append(gate[0]+1)
            pauli_strings.append([paulis, coeff.real, qubits])
    return pauli_strings

def group_pauli_strings(pauli_strings):
    groups = {}
    for string in pauli_strings:
        matched = False
        no_z_str = []
        no_z_qub = []
        for index in range(len(string[0])):
            if string[0][index] != Pauli.Z:
                no_z_str.append(string[0][index])
                no_z_qub.append(string[2][index])
        for group, strings in groups.items():
            if (tuple(no_z_str), tuple(no_z_qub)) == group:
                strings.append(string)
                matched = True
                break
        if not matched:
            groups[(tuple(no_z_str), tuple(no_z_qub))] = [string]
    return groups

def generate_two_det_state(occ1, occ2, nqubits):
    circ = Circuit(2*nqubits + 1, 1)
    circ.H(0)
    for index in occ1:
        circ.CX(0, index + 1)
    circ.X(0)
    for index in occ2:
        circ.CX(0, index + 1)
    return circ

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

def generate_excitation_strings(dets, ref, nspin_orb):
    excitation_strings = {}
    deexcitation_strings = {}
    ref_str = sum(2**r for r in ref)
    for det in dets:
        det_str = sum(2**d for d in det)
        excit = det_str ^ ref_str
        ex_str = ''
        deex_str = ''
        for index in range(nspin_orb):
            if 2**index & excit:
                if 2**index & ref_str:
                    ex_str += (str(index) + ' ')
                    deex_str += (str(index) + '^ ')
                else:
                    ex_str += (str(index) + '^ ')
                    deex_str += (str(index) + ' ')
        excitation_strings[det_str] = ex_str
        deexcitation_strings[det_str] = deex_str
    return excitation_strings, deexcitation_strings

def generate_excitation_operators(dets, ref, nspin_orb):
    ex, deex = generate_excitation_strings(dets, ref, nspin_orb)
    operators = [normal_ordered(FermionOperator(ex[i])) -
                 normal_ordered(FermionOperator(deex[i])) for i in ex]

    return operators

def bits_shared_ref(det, ref, nspin_orb):
    ref_str = sum(2**r for r in ref)
    det_str = sum(2**d for d in det)
    excit = det_str ^ ref_str
    shared_str = ''
    for index in range(nspin_orb):
        if 2**index & excit:
            if 2**index & ref_str:
                shared_str += str(index)
    return shared_str
def bits_not_shared_ref(det, ref, nspin_orb):
    ref_str = sum(2**r for r in ref)
    det_str = sum(2**d for d in det)
    excit = det_str ^ ref_str
    shared_str = ''
    for index in range(nspin_orb):
        if 2**index & excit:
            if not 2**index & ref_str:
                shared_str += str(index)
    return shared_str

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

def measure_hamil_overlap_shots(ps_g, circuits, n_shots):
    backend = AerBackend()
    hamil_elem = 0
    for pauli_group, circ in circuits.items():
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
    return hamil_elem
