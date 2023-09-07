''' Various overlap/operator measurement utility functions.'''

from typing import List

from pytket.pauli import Pauli
from pytket.extensions.qiskit import AerBackend
from openfermion.ops import FermionOperator, QubitOperator
#from openfermion import normal_ordered

import numpy as np

def count_set_bits(number:int):
    '''Count set bits in integer number.'''
    number_loc = number
    count = 0
    while number_loc:
        if number_loc & 1:
            count += 1
        number_loc = number_loc >> 1
    return count

def get_bits(number: int, nbits: int):
    ''' List the set and unset bits in the binary representation
        of integer number up to nbits. '''
    number_loc = number
    representation = []
    bits = 0
    while bits< nbits:
        if number_loc & 1:
            representation.append(1)
        else:
            representation.append(0)
        bits +=1
        number_loc = number_loc >> 1
    representation.reverse()
    representation = tuple(representation)
    return representation

def excit_to_det(occ_det: List[int], occ_ref:List[int]):
    ''' Find the sign s in a_ij...^ab... |D_0> = s |D_ij...^ab...>.'''
    det = sum(2**i for i in occ_det)
    ref = sum(2**i for i in occ_ref)
    excit = det^ref
    excit_level = count_set_bits(excit)/2
    nannih = excit_level
    ncreate = excit_level

    sign = 1

    for ibit in range(64):
        if ref & 2**ibit:
            if excit & 2**ibit:
                nannih -= 1
            elif (nannih+ncreate)%2 == 1:
                sign = -sign
        elif excit & 2**ibit:
            ncreate -= 1
    return sign

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
            pauli_strings.append([paulis, params[det]*coeff.imag/np.pi * 2, qubits, det])
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


def generate_excitation_strings(dets, ref, nspin_orb):
    '''Generate the excitation and deexcitation strings corresponding to
       some determinants, given a reference.'''
    excitation_strings = {}
    deexcitation_strings = {}
    ref_str = sum(2**r for r in ref)
    for det in dets:
        det_str = sum(2**d for d in det)
        excit = det_str ^ ref_str
        ex_str_cre = ''
        deex_str_cre = ''
        ex_str_anih = ''
        deex_str_anih = ''
        for index in range(nspin_orb):
            if 2**index & excit:
                if 2**index & ref_str:
                    ex_str_anih += (str(index) + ' ') 
                    deex_str_cre = (str(index) + '^ ') + deex_str_cre 
                else:
                    ex_str_cre = (str(index) + '^ ') + ex_str_cre
                    deex_str_anih += (str(index) + ' ') 
        #These should be reversed for some reason. Check for errors
        excitation_strings[det] = ex_str_cre + ex_str_anih
        deexcitation_strings[det] = deex_str_cre + deex_str_anih
    return excitation_strings, deexcitation_strings

def generate_excitation_operators(dets, ref, nspin_orb):
    '''Generate excitation operators from a set of determinants, given
       some reference.'''

    ex, deex = generate_excitation_strings(dets, ref, nspin_orb)
    operators = {i: (FermionOperator(ex[i])) -
                 (FermionOperator(deex[i])) for i in ex}
    print(operators)

    return operators

def bits_shared_ref(det, ref, nspin_orb):
    '''Find bits set in a determinant and in the reference.'''
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
    '''Find bits set in a determinant, but not in the reference.'''
    ref_str = sum(2**r for r in ref)
    det_str = sum(2**d for d in det)
    excit = det_str ^ ref_str
    shared_str = ''
    for index in range(nspin_orb):
        if 2**index & excit:
            if not 2**index & ref_str:
                shared_str += str(index)
    return shared_str

def generate_new_dets_list(dets, shots, nqubits, sym = None, sym_ref = 0):
    ''' Generate new determinant list after a spawning event.'''
    new = False
    no_ref = len([x for x in dets[0] if x%2==1])
    for shot in shots:
        det = tuple(i for i in range(nqubits) if shot[i] == 1)
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
    ''' Generate tuple used to order excitors in the tUCC ansatz.'''
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
    ''' Function to generate possible output states of a quantum circuit.
        Used to emulate spawning.'''#Probably should change/be somewhere else.
    backend = AerBackend()
    handle = backend.process_circuit(circ, n_shots = n_shots)
    counts = backend.get_result(handle).get_counts()
    return counts.keys()
