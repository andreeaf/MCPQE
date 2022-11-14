'''Module containing functions related to the implementation of UCC.'''


from pytket.circuit import Circuit, PauliExpBox, Qubit, Bit
from pytket.pauli import Pauli, QubitPauliString

from pytket.extensions.qiskit import AerStateBackend, AerBackend
import utils

def ucc_circuit(qubit_operator, params, nqubits):
    '''Generate corresponding UCC circuit from qubit operator'''

    pauli_strings = utils.pauli_strings_from_fermion_operators(qubit_operator, params)
    circ = Circuit(nqubits)

    for pauli_str in pauli_strings:
        circ.add_pauliexpbox(PauliExpBox(pauli_str[0], pauli_str[1]), pauli_str[2])
    return circ

def add_controlled_pauli_exp_box(circ, pauli_str, control):
    qubits = pauli_str[2]
    paulis = pauli_str[0]
    for index, qubit in enumerate(qubits):
        if paulis[index] == Pauli.X:
            circ.H(Qubit(qubit))
        if paulis[index] == Pauli.Y:
            circ.V(Qubit(qubit))
    for index, qubit in enumerate(qubits[:-1]):
        circ.CX(Qubit(qubit), Qubit(qubits[index+1]))
    circ.CRz(pauli_str[1], control, qubits[-1])
    for index in range(len(qubits)-1, 0, -1):
        circ.CX(Qubit(qubits[index-1]), Qubit(qubits[index]))
    for index, qubit in enumerate(qubits):
        if paulis[index] == Pauli.X:
            circ.H(Qubit(qubit))
        if paulis[index] == Pauli.Y:
            circ.Vdg(Qubit(qubit))
    return circ

def generate_overlap_measurement_circ(occ1, occ2, nqubits, operators, params):
    '''Generate overlap measurement circuit between U|D_1> and |D_2>'''

    ucc_circ = ucc_circuit(operators, params, nqubits)
    measurement = utils.generate_two_det_state(occ1, occ2, nqubits)
    for qubit in range(nqubits):
        measurement.CSWAP(0, qubit + 1, nqubits + 1 + qubit)
    measurement.add_circuit(ucc_circ, [Qubit(x + 1) for x in range(nqubits)])
    for qubit in range(nqubits):
        measurement.CSWAP(0, qubit + 1, nqubits + 1 + qubit)
    measurement.H(Qubit(0))
    #measurement.to_latex_file('abc.tex')
    return measurement

def generate_controlled_ucc_circ(nqubits, operators, params):
    circ = Circuit(nqubits + 1)
    pauli_strings = utils.pauli_strings_from_fermion_operators(operators, params, 1)
    for string in pauli_strings:
        circ = add_controlled_pauli_exp_box(circ, string, 0)
    return circ

def generate_controlled_overlap_measurement_circ(occ1, occ2, nqubits, ucc_circ):
    '''Generate overlap measurement circuit between U|D_1> and |D_2>'''
    measurement = Circuit(nqubits + 1, 1)
    measurement.H(0)
    for index in occ1:
        measurement.CX(0, index + 1)
    measurement.add_circuit(ucc_circ, measurement.qubits)
    measurement.X(0)
    for index in occ2:
        measurement.CX(0, index+1)
    measurement.H(Qubit(0))
    return measurement
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
            for pauli_str in pauli_gr:
                hamil_elem += pauli_str[1]*backend.get_pauli_expectation_value(comp_circ,
                                                 QubitPauliString([Qubit(x) for x in pauli_str[2]],
                                                 pauli_str[0]))
    return hamil_elem
