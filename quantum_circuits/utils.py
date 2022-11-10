from pytket.pauli import Pauli, QubitPauliString
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.circuit import Circuit, Qubit
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
        for group in groups:
            if (tuple(no_z_str), tuple(no_z_qub)) == group:
                groups[group].append(string)
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
    ref_str = sum([2**r for r in ref])
    for det in dets:
        det_str = sum([2**d for d in det])
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
    operators = [normal_ordered(FermionOperator(ex[i])) - normal_ordered(FermionOperator(deex[i])) for i in ex]

    return operators
