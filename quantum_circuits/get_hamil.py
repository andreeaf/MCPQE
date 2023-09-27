from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator, FermionOperator, normal_ordered, jordan_wigner
from openfermionpyscf import run_pyscf
import sys
sys.path.append('../')
import utils
from pytket.pauli import Pauli
from pytket.circuit import Circuit
import numpy as np
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
def get_hamil_matrix(hamil_pauli_strings: dict, nqubits: int):

    hmat_comp = np.zeros((2**(nqubits),2**(nqubits)), dtype=np.complex128)
    k = 0
    for string in hamil_pauli_strings:
        k +=1
        circ = Circuit(nqubits)
        for i in range(len(string[0])):
            if string[0][i] == Pauli.Z:
                circ.Z(string[2][i])
            if string[0][i] == Pauli.X:
                circ.X(string[2][i])
            if string[0][i] == Pauli.Y:
                circ.Y(string[2][i])
        mat = string[1] * circ.get_unitary()
        hmat_comp += mat
    hmat = hmat_comp.real
    return hmat
def transform_hmat(hmat, nqubits):
    new_hmat = np.zeros((2**(nqubits),2**(nqubits)))
    for i in range(2**nqubits):
        new_i = 0
        for k in range(nqubits):
            if i & 2**k:
                new_i += 2**(nqubits-1-k)
        for j in range(2**nqubits):
            new_j = 0
            for k in range(nqubits):
                if j & 2**k:
                    new_j += 2**(nqubits-1-k)
            new_hmat[new_i, new_j] = hmat[i,j]
    return new_hmat
for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    geometry = [['H', [0,0,0]], ['H', [0,0,x]], ['H', [0,0,2*x]], ['H', [0, 0, 3*x]]] 
    basis = 'sto-3g'
    multiplicity =1 
    charge = 0
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
#    molecule._canonical_orbitals[:,-1]*=-1
    #molecule._canonical_orbitals[:,1]*=-1
    print(molecule.canonical_orbitals)
    nqubits = 8
    print(molecule.hf_energy)
    print(molecule.fci_energy)
    molecule._one_body_integrals = None
    molecule._two_body_integrals = None
    e_hf = molecule.hf_energy
    ferm_h = get_fermion_operator(molecule.get_molecular_hamiltonian())
    qubit_h = jordan_wigner((ferm_h)) 
    ps_H = pauli_strings_from_fermion_operators_measurements([qubit_h])
    hmat = transform_hmat(get_hamil_matrix(ps_H, nqubits), nqubits)
    print(hmat[15,15])
    f = open('H4.'+str(x) +'.hmat', 'w')
    for i in range(2**nqubits):
        for j in range(2**nqubits):
            f.write("{:.8f}".format(hmat[i,j]) + '   ')
        f.write('\n')
    f.close()

