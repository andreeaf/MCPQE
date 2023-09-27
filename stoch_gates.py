'''Module for noise gate introduction.'''
from pytket.circuit import Circuit, Qubit, OpType, Bit
import numpy as np
from sympy import Symbol
import utils
import encode_ucc_wavefunction as euw
def noise_x(phi, circ, qubit):
    '''Add a parametrised gate to qubit in circuit circ. When phi = 0, the
    gate is I, when phi = pi the gate is X.'''
    circ.Ry(phi/np.pi,qubit)
    circ.Rz(2*phi/np.pi,qubit)
    return circ

def noise_y(phi, circ, qubit):
    '''Add a parametrised gate to qubit in circuit circ. When phi = 0, the
    gate is I, when phi = pi the gate is Y.'''
    circ.Rx(phi/np.pi,qubit)
    circ.Rz(2*phi/np.pi,qubit)
    return circ

def noise_z(phi, circ, qubit):
    '''Add a parametrised gate to qubit in circuit circ. When phi = 0, the
    gate is I, when phi = pi the gate is Z.'''
    circ.Rx(phi/np.pi,qubit)
    circ.Ry(phi/np.pi,qubit)
    circ.add_phase(3/2*phi/np.pi)
    return circ

def noise_depolarisation(phis, circ, qubit):
    '''Add a stochastic X, Y and Z gate to a circuit to model depolarisation
    noise.'''
    noise_x(phis[0], circ, qubit)
    noise_y(phis[1], circ, qubit)
    noise_z(phis[2], circ, qubit)

def amplitude_damping_noise(gamma, circ, qubit, ancilla):
    '''Add an amplitude damping gate to a circuit. Currently difficult to
    implement, cannot get_unitary with Reset in pytket. Might need backend.'''
    circ.add_gate(OpType.Reset, [ancilla])
    circ.CRy(gamma, Qubit(qubit), ancilla)
    circ.CX(ancilla, Qubit(qubit))
    circ.Measure(ancilla, Bit(0))
    return circ

def generate_noisy_controlled_ucc_circ(nqubits, operators, params,noise_symbols,
                                       depolarising_noise=True, amplitude_damping=True):
    '''Given a set of unitary excitation operators and parameters, generate
    the corresponding, potentially noisy, controlled UCC circuit.'''
    circ = Circuit(nqubits + 1)
    anc = Qubit('ancilla')
    circ.add_qubit(anc)
    circ.add_bit(Bit(0))
    pauli_strings = utils.pauli_strings_from_fermion_operators(operators, params, 1)
    noise_symbols2 = {}
    for string in pauli_strings:
        circ = euw.add_controlled_pauli_exp_box(circ, string, 0)
        noise_symbols[string[3]] = []
        noise_symbols2[string[3]] = []

        for qubit in string[2]:
            if depolarising_noise:
                symbols = [Symbol('phi'+str(string[3]) + str(len(noise_symbols[string[3]]))),
                           Symbol('phi'+str(string[3]) + str(len(noise_symbols[string[3]])+1)),
                           Symbol('phi'+str(string[3]) + str(len(noise_symbols[string[3]])+2))]
                noise_symbols[string[3]] += symbols
                noise_depolarisation(symbols, circ, qubit)
            if amplitude_damping:
                symbols = [Symbol('gamma'+str(string[3])+ str(len(noise_symbols2[string[3]])))]
                noise_symbols2[string[3]] += symbols
                amplitude_damping_noise(symbols[0], circ, qubit, anc)
            noise_symbols[string[3]] += noise_symbols2[string[3]]
    return circ

def get_noise(noise_symbols, prob, gamma):
    '''Given a depolarisation probability p and an amplitude damping rate gamma,
    generate the corresponding noise_symbol:angle mapping.'''
    maps = {}
    if gamma > 0 and prob > 0:
        for i in range(int(len(noise_symbols)/4)):
            rand = np.random.random()

            if rand > prob:
                maps.update({noise_symbols[4*i]:0, noise_symbols[4*i +1]:0, noise_symbols[4*i+2]:0})
            elif rand < prob/3:
                maps.update({noise_symbols[4*i]:np.pi, noise_symbols[4*i +1]:0,
                            noise_symbols[4*i+2]:0})
            elif rand < 2*prob/3:
                maps.update({noise_symbols[4*i]:0, noise_symbols[4*i +1]:np.pi,
                             noise_symbols[4*i+2]:0})
            else:
                maps.update({noise_symbols[4*i]:0, noise_symbols[4*i +1]:0,
                             noise_symbols[4*i+2]:np.pi})
            noise_symbols[4*i + 3] = gamma
    elif prob > 0:
        for i in range(int(len(noise_symbols)/3)):
            rand = np.random.random()

            if rand > prob:
                maps.update({noise_symbols[3*i]:0, noise_symbols[3*i +1]:0,
                             noise_symbols[3*i+2]:0})
            elif rand < prob/3:
                maps.update({noise_symbols[3*i]:np.pi, noise_symbols[3*i +1]:0,
                             noise_symbols[3*i+2]:0})
            elif rand < 2*prob/3:
                maps.update({noise_symbols[3*i]:0, noise_symbols[3*i +1]:np.pi,
                             noise_symbols[3*i+2]:0})
            else:
                maps.update({noise_symbols[3*i]:0, noise_symbols[3*i +1]:0,
                             noise_symbols[3*i+2]:np.pi})
    elif gamma > 0:
        for i in range(len(noise_symbols)):
            noise_symbols[i] = gamma
    return maps
