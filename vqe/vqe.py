from typing import Union
from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator, FermionOperator, normal_ordered
from openfermionpyscf import run_pyscf
from pytket.pauli import Pauli
from pytket.circuit import Circuit, PauliExpBox
from sympy import Symbol
import numpy as np
import scipy
from qiskit.algorithms.optimizers import SPSA
from pytket.passes import DecomposeBoxes
from scipy.sparse import coo_array, csc_array, save_npz
import bisect


def apply_H(terms, qub):
    new_terms = {}
    for term in terms:
        if term & 2**qub:
            if term in new_terms:
                new_terms[term] += -terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] += terms[term]/np.sqrt(2)
            else:
                new_terms[term] = -terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] = terms[term]/np.sqrt(2)
        else:
            if term in new_terms:
                new_terms[term] += terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] += terms[term]/np.sqrt(2)
            else:
                new_terms[term] = terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] = terms[term]/np.sqrt(2)
    return new_terms

def apply_V(terms, qub):
    new_terms = {}
    for term in terms:
        if term & 2**qub:
            if term in new_terms:
                new_terms[term] += terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] += -1j*terms[term]/np.sqrt(2)
            else:
                new_terms[term] = terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] = -1j*terms[term]/np.sqrt(2)
        else:
            if term in new_terms:
                new_terms[term] += terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] += -1j*terms[term]/np.sqrt(2)
            else:
                new_terms[term] = terms[term]/np.sqrt(2)
                new_terms[term^(2**qub)] = -1j*terms[term]/np.sqrt(2)
    return new_terms
def apply_Z(terms, qub):
    new_terms = {}
    for term in terms:
        if term & 2**qub:
            if term in new_terms:
                new_terms[term] -= terms[term]
            else:
                new_terms[term] = -terms[term]
        else:
            if term in new_terms:
                new_terms[term] += terms[term]
            else:
                new_terms[term] = terms[term]
    return new_terms
def apply_X(terms, qub):
    new_terms = {}
    for term in terms:
        if term in new_terms:
            new_terms[term^(2**qub)] += terms[term]
        else:
            new_terms[term^(2**qub)] = terms[term]
    return new_terms
def apply_Y(terms, qub):
    new_terms = {}
    for term in terms:
        if term & 2**qub:
            if term in new_terms:
                new_terms[term^(2**qub)] += -1j*terms[term]
            else:
                new_terms[term^(2**qub)] = -1j*terms[term]
        else:
            if term in new_terms:
                new_terms[term^(2**qub)] += 1j*terms[term]
            else:
                new_terms[term^(2**qub)] = 1j*terms[term]
    return new_terms
def get_hamil_and_overlap_matrix_csc(hamil_pauli_strings: dict, nqubits: int, filename:str):

    ''' Function to generate Hamiltonian and overlap matrices for Hadamard-test-type
        measurement circuit with 1 ancilla. To measure <Psi_k|O|Psi_l> the Paulis
        corresponding to O must be applied to the system qubits and a Z gate to the
        ancilla. This is included in the Pauli_strings. '''

    hmat_comp = csc_array((2**(nqubits),2**(nqubits)), dtype=np.complex128)
    k = 0
    mapping = {x:nqubits-1-x for x in range(nqubits)}
    for pauli_gr in hamil_pauli_strings:
        k +=1
        print(k, len(hamil_pauli_strings))
        rows = []
        columns = []
        vals = [] 
        for i in range(2**(nqubits)):
            terms = {i:1.0}
            for index in range(len(pauli_gr[0])):
                if pauli_gr[0][index] == Pauli.X:
                    terms = apply_X(terms, mapping[pauli_gr[2][index]])         
                elif pauli_gr[0][index] == Pauli.Y:
                    terms = apply_Y(terms, mapping[pauli_gr[2][index]])         
                elif pauli_gr[0][index] == Pauli.Z:
                    terms = apply_Z(terms, mapping[pauli_gr[2][index]])         
            for term in terms:
                rows.append(i)
                columns.append(term)
                vals.append(terms[term])
        hmat_comp += pauli_gr[1]*coo_array((vals, (rows, columns))).tocsc() 
    
    rows = []
    columns = []
    vals = [] 
    for i in range(2**(nqubits)):
        terms = {i:1.0}
        for term in terms:
            rows.append(i)
            columns.append(term)
            vals.append(terms[term])
    zmat_comp = coo_array((vals, (rows, columns))).tocsc()
    zmat = zmat_comp.real
    hmat = hmat_comp.real
    save_npz(filename+'hmat2', hmat)
    save_npz(filename+'zmat2', zmat)
    return hmat, zmat
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

def measure_hamil_overlap_groups_matrices(ps_g, mats, mcirc, mapping, nshots = 0, gaussian = False):
    hamil_elem = 0
    k = 0
    for pauli_group, mat in mats.items():
        k += 1
        if k%100 == 0:
            print(k, len(mats))
        if pauli_group in ps_g:
            circ = mcirc.copy()
            circ.append(mat)
            vector = circ.get_statevector()
            #vector = mat.dot(mcirc.get_statevector())
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
def get_hamil_group_matrices_csc(ps_g, nqubits):
    mats = {}
    mapping = {x:nqubits-1-x for x in range(nqubits)}
    k = 1
    for pauli_gr in ps_g:
        print(k, len(ps_g))    
        rows = []
        columns = []
        vals = [] 
        for i in range(2**nqubits):
            terms = {i:1.0}
            for index in range(len(pauli_gr[0])):
                if pauli_gr[0][index] == Pauli.X:
                    terms = apply_H(terms, mapping[pauli_gr[1][index]])         
                elif pauli_gr[0][index] == Pauli.Y:
                    terms = apply_V(terms, mapping[pauli_gr[1][index]])         
            for term in terms:
                rows.append(i)
                columns.append(term)
                vals.append(terms[term])
        mats[pauli_gr] = coo_array((vals, (rows, columns))).tocsc() #csr_matrix(measurement_circ.get_unitary())
        k += 1
    return mats

def measure_hamil_overlap_groups_matrices_csc(ps_g, mats, mcirc, mapping, nshots = 0, gaussian = False):
    hamil_elem = 0
    sv = mcirc.get_statevector()
    for pauli_group, mat in mats.items():
        if pauli_group in ps_g:
            vector = mat.dot(sv)
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
def get_counts(vector, nshots, mapping, gaussian = False):
    vector_square = np.conj(vector)*vector
    distribution = np.zeros(len(vector))
    prob = 0
    for i in range(len(vector_square)):
        if vector_square[i] > 1e-9:
            prob += vector_square[i]
            last_dist = i
        distribution[i] = prob
    distribution[:]/=distribution[last_dist]
    if nshots == 0 or gaussian:
        counts = {mapping[a]:vector_square[a].real for a in mapping}
        return counts
    counts = {}
    rands = np.random.rand(nshots)
    length = len(distribution)
    rang = range(length)
    for rand in rands:
        j = bisect.bisect_left(distribution, rand)
        if mapping[j] in counts:
            counts[mapping[j]] +=1
        else:
            counts[mapping[j]] =1
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
    k = 1
    for pauli_gr in ps_g:
        print(k, len(ps_g))    
        measurement_circ = Circuit(nqubits)
        for index in range(len(pauli_gr[0])):
            if pauli_gr[0][index] == Pauli.X:
                measurement_circ.H(pauli_gr[1][index])
            if pauli_gr[0][index] == Pauli.Y:
                measurement_circ.V(pauli_gr[1][index])
        mats[pauli_gr] = measurement_circ #csr_matrix(measurement_circ.get_unitary())
        k += 1
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
    if (n_eval % 100) == 0:
        print('interm', n_eval)
    pops_loc = {dets[i]:pops[i] for i in range(len(dets))}
    param_map = {params[d]:pops_loc[d]/pops_loc[ref] for d in params}

    current_circ = circ.copy()

    current_circ.symbol_substitution(param_map)
    
    energy = measure_hamil_overlap_groups_matrices_csc(groups, mats, current_circ, mapping, nshots = shots, gaussian = False)
    return energy
def cost_function2(pops, dets, params, circ, groups, hmat, mapping, epsilon):
    global n_eval
    n_eval += 1
    pops_loc = {dets[i]:pops[i] for i in range(len(dets))}
    param_map = {params[d]:pops_loc[d]/pops_loc[ref] for d in params}

    current_circ = circ.copy()

    current_circ.symbol_substitution(param_map)
    
    energy = (current_circ.get_statevector()).dot(hmat.dot(current_circ.get_statevector())).real+ np.random.normal(0, epsilon)
    return energy
    
#ff = open('H4-VQE-sv-LBFGS.dat', 'w')    
#f = open('H4-VQE-COBYLA-1e4-1e-4err.dat', 'w')    
#ffff = open('H4-VQE-CG-1e4-1e-4err.dat', 'w')
for r in [3.0]:
    geometry = [['H', [0,0,0]], ['H', [0,0,r]], ['H', [0,0,2*r]], ['H', [0,0,3*r]], ['H', [0,0,4*r]], ['H', [0,0,5*r]], ['H', [0,0,6*r]], ['H', [0,0,7*r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    nqubits = 16 
    dets = np.genfromtxt('H8_DETS', dtype=int)
    for i in range(len(dets)):
        for j in range(8):
            dets[i][j] -= 1
    dets = [tuple(x) for x in dets]
    #print(dets)
    mapping = {det: get_bits(det, nqubits) for det in range(2**(nqubits))}
    hf_e, fci_e, qubit_h = get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    ps_H = get_hamil_pauli_strings(qubit_h)
    groups = group_pauli_strings(ps_H)
    #mats = get_hamil_group_matrices_csc(groups, nqubits)
    #hmat, zmat = get_hamil_and_overlap_matrix_csc(ps_H, nqubits, 'H6'+str(r))
    ref = (0,1,2,3,4,5,6,7)
    dets.append(ref)
    #dets = [(1, 2, 3, 4), (2, 3, 4, 5), (1, 3, 4, 6), (1, 2, 5, 6), (1, 2, 4, 7), (2, 3, 6, 7), (0, 2, 3, 5), (0, 3, 5, 6), (0, 3, 4, 7), (0, 2, 5, 7), (0, 1, 3, 6), (0, 1, 4, 5), (0, 1, 6, 7), (0, 1, 2, 7), (0, 1, 2, 3)]
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
    circ2 = circ.copy()
    DecomposeBoxes().apply(circ2)
    print('N2', circ2.n_2qb_gates()) 
    exit()
    pops = {det:0 for det in dets}
    pops[ref] = 1
    for repeat in range(5):
        fff = open('H8-VQE-SPSA-1e4-1000it-hmat5.dat', 'a')    
        print(r)
        
        pops = [1]+[0]*(len(dets)-1)
        shots = 0
        n_eval = 0
    
        def cost_function3(pops):
            return cost_function2(pops, dets, params, circ, groups, hmat, mapping, epsilon)
    
     #   res = scipy.optimize.minimize(cost_function, pops, args = (dets, params, circ, groups, mats, mapping, shots), method = 'L-BFGS-B', tol = 1e-6, options = {'disp':True, 'maxiter':1000})
    #    print('Statevector')
    #    print(n_eval)
    #    print(res.x)
    #    print(res.fun)
    #    ff.write(str(r) + '     '+ str(res.fun) + '    ' +str(n_eval) + '    ' + str(res.nit)+'    '+ str(len(mats.items()))+ str('\n'))
    #   pops = [1]+[0]*(len(dets)-1)
        shots = 10000
        epsilon = 1/shots
        epsilon *= np.sqrt(sum(x[1]**2 for x in ps_H))
     #   n_eval = 0
     #   res = scipy.optimize.minimize(cost_function, pops, args = (dets, params, circ, groups, mats, mapping, shots), method = 'COBYLA', tol = 1e-4, options = {'disp':True, 'maxiter':1000})
     #   print('Shots')
     #   print(n_eval)
     #   print(res.x)
     #   print(res.fun)
     #   print(res.keys())
     #   f.write(str(r) + '    ' + str(fci_e) +'    ' + str(hf_e)+'     '+ str(res.fun) + '    ' +str(n_eval) + '    ' + str(len(mats.items()))+ '\n')
     #   n_eval = 0
     #   pops = [1]+[0]*(len(dets)-1)
     #   res = scipy.optimize.minimize(cost_function, pops, args = (dets, params, circ, groups, mats, mapping, shots), method = 'CG', tol = 1e-4, options = {'disp':True, 'maxiter':1000, "return_all": True})
     #   print('Shots')
     #   print(n_eval)
     #   print(res.x)
     #   print(res.fun)
     #   print(res.keys())
     #   ffff.write(str(r) + '    ' + str(fci_e) +'    ' + str(hf_e)+'     '+ str(res.fun) + '    ' +str(n_eval) + '    ' + str(res.nit) + '    ' + str(len(mats.items()))+ '\n')
        pops = [1]+[0]*(len(dets)-1)
        spsa = SPSA(maxiter=1000)
        n_eval = 0
        res = spsa.minimize(cost_function3, pops)
        print(res.x)
        print(res.fun)
        print(n_eval)
        fff.write(str(r) + '    ' + str(fci_e) +'    ' + str(hf_e)+'     '+ str(res.fun) + '    ' +str(n_eval) + '    '+ str(res.nit) + '    ' + '\n')
        fff.close()
    #ffff.close()
    #ff.close()
    #f.close()
