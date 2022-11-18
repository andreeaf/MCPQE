''' Various overlap/operator measurement utility functions.'''

from pytket.pauli import Pauli, QubitPauliString
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.circuit import Qubit, Bit
from openfermion.transforms import normal_ordered
from openfermion.ops import FermionOperator, QubitOperator

import numpy as np

def pauli_strings_from_fermion_operators(operators: dict,
                                         params: dict,
                                         ancilla: int = 0):
    '''Function to get parametrised Pauli strings from qubit operators and their
    IN:
    corresponding parameters.
    operators: dict of operators corresponding to excitations to a given
               determinant.
    params: dict of corresponding coefficients.
    ancilla: number of ancilla qubits the circuit will have - for correct qubit
              labels.
    Returns:
    pauli_strings: list of all parametrised Pauli strings.
    '''
    pauli_gate = {'X': Pauli.X, 'Y': Pauli.Y, 'Z': Pauli.Z, 'I': Pauli.I}
    pauli_strings = []
    for det, operator in operators.items():
        for term, coeff in operator.terms.items():
            paulis = []
            qubits = []
            for gate in term:
                paulis.append(pauli_gate[gate[1]])
                qubits.append(gate[0] + ancilla)
            pauli_strings.append([paulis, params[det]*coeff.imag/np.pi*2, qubits])
    return pauli_strings


def pauli_strings_from_fermion_operators_measurements(operators: list[QubitOperator]):
    '''Function to get Pauli strings corresponding to operators to be measured
    in a circuit with one ancilla qubit on which a Z operation is measured.
    IN:
    operators: list of QubitOperators
    Returns:
    pauli_strings: list of all parametrised Pauli strings
    '''
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

def group_pauli_strings(pauli_strings: list):
    '''Function to group Pauli strings that require the same measurement circuit.
    IN:
    pauli_strings:list of all parametrised Pauli strings
    Returns:
    groups:list of all groups of parametrised Pauli strings
    '''
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
        excitation_strings[det] = ex_str
        deexcitation_strings[det] = deex_str
    return excitation_strings, deexcitation_strings

def generate_excitation_operators(dets, ref, nspin_orb):
    ex, deex = generate_excitation_strings(dets, ref, nspin_orb)
    operators = {i: normal_ordered(FermionOperator(ex[i])) -
                 normal_ordered(FermionOperator(deex[i])) for i in ex}

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
    all_counts = []
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
        for count in counts:
            if count not in all_counts:
                all_counts.append(count)
    return hamil_elem, all_counts

def generate_new_dets_list(dets, shots, nqubits, sym = None, sym_ref = 0):
    new = False
    no_ref = len([x for x in dets[0] if x%2==1])
    for shot in shots:
        det = tuple([i for i in range(nqubits) if shot[i] == 1])
        diff = 0
        for orb in det:
            if orb not in dets[0]:
                diff +=1
        symm = True
        if sym is not None:
            det_sym = 0
            for i in det:
               det_sym  ^= sym[i]
            if det_sym != sym_ref:
                symm = False
        num_odd = len([x for x in det if x%2==1])
        if det not in dets and diff<=2 and len(det) == len(dets[0]) and symm and num_odd==no_ref:
            dets.append(det)
            new = True
    return dets, new

def dets_list_order(det, ref):
    if det == ref:
        return (0, 0, 0)
    for orb in ref:
        if orb not in det:
            major = orb
            break
    count = 0
    for orb in ref:
        if orb not in det:
            count += 1
    minor = sum(2**d for d in det)
    return (major, count, minor)

def get_spawns(circ, n_shots=10):

    backend = AerBackend()
    handle = backend.process_circuit(circ, n_shots = n_shots)
    counts = backend.get_result(handle).get_counts()
    return counts.keys()
