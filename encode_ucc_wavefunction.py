'''Module containing functions related to the implementation of UCC.'''

from typing import Union

from pytket.circuit import Circuit, PauliExpBox, Qubit
from pytket.pauli import Pauli

import utils

def ucc_circuit(qubit_operator, params, nqubits):
    '''Generate corresponding UCC circuit from qubit operator'''

    pauli_strings = utils.pauli_strings_from_fermion_operators(qubit_operator, params)
    circ = Circuit(nqubits)

    for pauli_str in pauli_strings:
        circ.add_pauliexpbox(PauliExpBox(pauli_str[0], -pauli_str[1]), pauli_str[2])
    return circ

def add_controlled_pauli_exp_box(circ:Circuit, pauli_str:list, control:Union[int, Qubit]):
    ''' Add a controlled Pauli gadget to a circuit.
    IN:
        circ: current circuit
        pauli_str: Pauli string to encode in exp_box
        control: control qubit'''
    qubits = pauli_str[2]
    paulis = pauli_str[0]
    for index, qubit in enumerate(qubits):
        if paulis[index] == Pauli.X:
            circ.H(Qubit(qubit))
        if paulis[index] == Pauli.Y:
            circ.V(Qubit(qubit))
    for index, qubit in enumerate(qubits[:-1]):
        circ.CX(Qubit(qubit), Qubit(qubits[index+1]))
    circ.CRz(-pauli_str[1], control, qubits[-1])
    for index in range(len(qubits)-1, 0, -1):
        circ.CX(Qubit(qubits[index-1]), Qubit(qubits[index]))
    for index, qubit in enumerate(qubits):
        if paulis[index] == Pauli.X:
            circ.H(Qubit(qubit))
        if paulis[index] == Pauli.Y:
            circ.Vdg(Qubit(qubit))
    return circ

def generate_ucc_state(ref:list,
                       operators:dict,
                       params:dict,
                       nqubits:int):
    '''Generate a circuit representation of the UCC wavefunctio U|D_0>.
    IN:
        ref: occupation string representation of the reference determinant.
        operators: excitation operators
        params: associated amplitudes
        nqubits: number of qubits'''

    ucc_circ = ucc_circuit(operators, params, nqubits)
    circ = Circuit(nqubits)
    for occ in ref:
        circ.X(occ)
    circ.add_circuit(ucc_circ, [Qubit(x) for x in range(nqubits)])
    return circ

def generate_controlled_ucc_circ(nqubits:int,
                                 operators:dict,
                                 params:dict):
    '''Generate corresponding controlled UCC circuit from qubit operator'''
    circ = Circuit(nqubits + 1)
    pauli_strings = utils.pauli_strings_from_fermion_operators(operators, params, 1)
    for string in pauli_strings:
        circ = add_controlled_pauli_exp_box(circ, string, 0)
    return circ

def generate_controlled_overlap_measurement_circ(occ1, occ2, nqubits, ucc_circ):
    '''Generate overlap measurement circuit between U|D_1> and |D_2>'''
    measurement = Circuit(nqubits + 1, 1)
    measurement.H(0)
    measurement.X(0)
    for index in occ1:
        measurement.CX(0, index + 1)
    measurement.add_circuit(ucc_circ, measurement.qubits)
    measurement.X(0)
    for index in occ2:
        measurement.CX(0, index+1)
    measurement.H(0)
    return measurement

def generate_controlled_overlap_measurement_circ_trial(occ1, occ2, nqubits, ucc_circ, ucc_circ2):
    '''Generate overlap measurement circuit between U|D_1> and |D_2>'''
    measurement = Circuit(nqubits + 1, 1)
    measurement.H(0)
    measurement.X(0)
    for index in occ1:
        measurement.CX(0, index + 1)
    measurement.add_circuit(ucc_circ, measurement.qubits)
    measurement.X(0)
    for index in occ2:
        measurement.CX(0, index+1)
    measurement.add_circuit(ucc_circ2, measurement.qubits)
    measurement.H(0)
    return measurement
