from typing import Union
from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator, FermionOperator, normal_ordered
from openfermionpyscf import run_pyscf
from pytket.pauli import Pauli
from pytket.circuit import Circuit, PauliExpBox
from sympy import Symbol
import numpy as np
import scipy
def ucc_circuit(qubit_operator, params, nqubits):
    '''Generate corresponding UCC circuit from qubit operator'''

    pauli_strings = pauli_strings_from_fermion_operators(qubit_operator, params)
    circ = Circuit(nqubits)

    for pauli_str in pauli_strings:
        circ.add_pauliexpbox(PauliExpBox(pauli_str[0], pauli_str[1]), pauli_str[2])
    return circ
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

def measure_hamil_overlap_groups_matrices(ps_g, mats, mcirc, mapping, nshots = 0):
    hamil_elem = 0
    k = 0
    for pauli_group, mat in mats.items():
        if pauli_group in ps_g:
            vector = np.dot(mat, mcirc.get_statevector().T)
            counts = get_counts(vector, nshots, mapping)
            #print(k)
            k +=1
            for pauli_str in ps_g[pauli_group]:
                avg = 0
                for count in counts:
                    sign = 1
                    for index in pauli_str[2]:
                        if count[index] == 1:
                            sign *= -1
                    avg += sign * counts[count]
                if nshots:
                    hamil_elem += pauli_str[1] * avg/nshots
                else:
                    hamil_elem += pauli_str[1]*avg
    return hamil_elem

def get_counts(vector, nshots, mapping):
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
    if nshots == 0:
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
def get_qubit_molecular_hamiltonian(geometry: list[str, list[int]],
                                    basis: str,
                                    multiplicity: int,
                                    charge: int,
                                    stoch: bool = False):
    ''' Function to generate a qubit hamiltonian from a molecular input, using OpenFermion.
    IN:
        geometry: List of atoms and coordinates.
        basis: Basis set.
        multiplicity: 2S+1.
        charge: molecular charge.
    Returns:
        qubit_h: OpenFermion QubitOperator corresponding to the molecular hamiltonian.'''

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    if stoch:
        qubit_h = []
        ferm_h = get_fermion_operator(molecule.get_molecular_hamiltonian())
        for term, coeff in ferm_h.terms.items():
            qubit_h.append([jordan_wigner(FermionOperator(term, coeff)), coeff])
        print(molecule.fci_energy)
        print(molecule.hf_energy)
        return molecule.hf_energy, molecule.fci_energy, qubit_h
        
    qubit_h = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    print(molecule.fci_energy)
    print(molecule.hf_energy)
    return molecule.hf_energy, molecule.fci_energy, qubit_h

def get_hamil_pauli_strings(qubit_h: Union[QubitOperator, list[QubitOperator]]):

    ''' Function to generate a list of parametrised Pauli strings corresponding
        to a qubit operator.'''

    if isinstance(qubit_h, QubitOperator):
        return pauli_strings_from_fermion_operators_measurements([qubit_h])
    else:
        return pauli_strings_from_fermion_operators_measurements([x[0] for x in qubit_h])
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
            paulis = []
            qubits = []
            for gate in term:
                paulis.append(pauli_gate[gate[1]])
                qubits.append(gate[0])
            pauli_strings.append([paulis, coeff.real, qubits])
    return pauli_strings

def get_hamil_group_matrices(ps_g, nqubits):
    mats = {}
    for pauli_gr in ps_g:
        measurement_circ = Circuit(nqubits)
        for index in range(len(pauli_gr[0])):
            if pauli_gr[0][index] == Pauli.X:
                measurement_circ.H(pauli_gr[1][index])
            if pauli_gr[0][index] == Pauli.Y:
                measurement_circ.V(pauli_gr[1][index])
        mats[pauli_gr] = measurement_circ.get_unitary()
    return mats

def get_bits(n, nbits):
    m = n
    rep = []
    bits = 0
    while bits< nbits:
        if m & 1:
            rep.append(1)
        else:
            rep.append(0)
        bits +=1
        m = m >> 1
    rep.reverse()
    rep = tuple(rep)
    return rep
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
                else:
                    ex_str += (str(index) + '^ ')
        for index in range(nspin_orb-1, -1, -1):
            if 2**index & excit:
                if 2**index & ref_str:
                    deex_str += (str(index) + '^ ')
                else:
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

def cost_function(pops, dets, params, circ, groups, mats, mapping, shots):
    global n_eval
    n_eval += 1
    pops_loc = {dets[i]:pops[i] for i in range(len(dets))}
    param_map = {params[d]:pops_loc[d]/pops_loc[ref] for d in params}

    current_circ = circ.copy()

    current_circ.symbol_substitution(param_map)
    
    energy = measure_hamil_overlap_groups_matrices(groups, mats, current_circ, mapping, nshots = shots)
    return energy
    
ff = open('H4-VQE-sv.dat', 'w')    
f = open('H4-VQE-shots.dat', 'w')    
for r in np.linspace(1.00,3.50,26):
    print(r)
    geometry = [['H', [0,0,0]], ['H', [0,0,r]], ['H', [0,0,2*r]], ['H', [0,0,3*r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    nqubits = 8
    mapping = {det: get_bits(det, nqubits) for det in range(2**(nqubits))}
    hf_e, fci_e, qubit_h = get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    ps_H = get_hamil_pauli_strings(qubit_h)
    groups = group_pauli_strings(ps_H)
    mats = get_hamil_group_matrices(groups, nqubits)
    dets = [(1, 2, 3, 4), (2, 3, 4, 5), (1, 3, 4, 6), (1, 2, 5, 6), (1, 2, 4, 7), (2, 3, 6, 7), (0, 2, 3, 5), (0, 3, 5, 6), (0, 3, 4, 7), (0, 2, 5, 7), (0, 1, 3, 6), (0, 1, 4, 5), (0, 1, 6, 7), (0, 1, 2, 7), (0, 1, 2, 3)]
    ref = (0,1,2,3)
    #dets = [(0,1,2), (0,3,4), (2,4,5), (0,2,5), (1,2,4)]
    #dets = [(0,1), (2,3), (4,5), (0,5), (1,4)]
    
    dets.sort(key = lambda x: dets_list_order(x, ref))
    operators = generate_excitation_operators(dets[1:], ref, nqubits)
    qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
    params = {d:Symbol('t'+str(bits_shared_ref(d,ref,nqubits))+'_'+str(bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
    circ = Circuit(nqubits)
    for q in ref:
        circ.X(q)
    
    ucc_circ = ucc_circuit(qubit_ops, params, nqubits)
    circ.append(ucc_circ)
    
    pops = {det:0 for det in dets}
    pops[ref] = 1
    
    pops = [1]+[0]*(len(dets)-1)
    shots = 0
    n_eval = 0
    res = scipy.optimize.minimize(cost_function, pops, args = (dets, params, circ, groups, mats, mapping, shots), method = 'L-BFGS-B', tol = 1e-3, options = {'disp':True, 'maxiter':1000})
    print('Statevector')
    print(n_eval)
    print(res.x)
    print(cost_function(res.x, dets, params, circ, groups, mats, mapping, shots))
    ff.write(str(r) + '     '+ str(cost_function(res.x, dets, params, circ, groups, mats, mapping, shots)) + '    ' +str(n_eval) + '    ' + str(len(mats.items()))+ str('\n'))
    pops = [1]+[0]*(len(dets)-1)
    shots = 10000
    n_eval = 0
    res = scipy.optimize.minimize(cost_function, pops, args = (dets, params, circ, groups, mats, mapping, shots), method = 'COBYLA', tol = 1e-3, options = {'disp':True, 'maxiter':1000})
    print('Shots')
    print(n_eval)
    print(res.x)
    print(cost_function(res.x, dets, params, circ, groups, mats, mapping, shots))
    f.write(str(r) + '    ' + str(fci_e) +'    ' + str(hf_e)+'     '+ str(cost_function(res.x, dets, params, circ, groups, mats, mapping, shots)) + '    ' +str(n_eval)  + '    ' + str(len(mats.items()))+ '\n')
ff.close()
f.close()
