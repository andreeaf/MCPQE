'''Hamiltonian manipulation module'''

from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator, FermionOperator
from openfermionpyscf import run_pyscf
from typing import Union
import utils
import numpy as np
from pytket.circuit import Circuit
from pytket.pauli import Pauli
from scipy.sparse import csr_matrix, save_npz
import pickle


def get_qubit_molecular_hamiltonian(geometry: list[str, list[int]],
                                    basis: str,
                                    multiplicity: int,
                                    charge: int,
                                    ncore: int = 0):
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
    occupied_indices = list(range(ncore))
        
    if ncore:
        qubit_h = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian(occupied_indices=occupied_indices)))
    else:
        qubit_h = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    
    print('FCI Energy: ', molecule.fci_energy)
    print('HF Energy: ', molecule.hf_energy)
    return molecule.hf_energy, molecule.fci_energy, qubit_h

def get_qubit_molecular_hamiltonian_squared(geometry: list[str, list[int]],
                                    basis: str,
                                    multiplicity: int,
                                    charge: int,
                                    omega: float = 0):
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
        
    ferm_hamil = get_fermion_operator(molecule.get_molecular_hamiltonian()) - (molecule.hf_energy+omega) *FermionOperator('')
    qubit_h = jordan_wigner(ferm_hamil * ferm_hamil)
    print('FCI Energy: ', molecule.fci_energy)
    print('HF Energy: ', molecule.hf_energy)
    return molecule.hf_energy, molecule.fci_energy, qubit_h

def get_hamil_pauli_strings(qubit_h: Union[QubitOperator, list[QubitOperator]]):

    ''' Function to generate a list of parametrised Pauli strings corresponding
        to a qubit operator or list of Pauli operators.''' 

    if isinstance(qubit_h, QubitOperator):
        return utils.pauli_strings_from_fermion_operators_measurements([qubit_h])
    else:
        return utils.pauli_strings_from_fermion_operators_measurements([x[0] for x in qubit_h])

def get_hamil_and_overlap_matrix(hamil_pauli_strings: dict, nqubits: int, filename:str):

    ''' Function to generate Hamiltonian and overlap matrices for Hadamard-test-type
        measurement circuit with 1 ancilla. To measure <Psi_k|O|Psi_l> the Paulis
        corresponding to O must be applied to the system qubits and a Z gate to the
        ancilla. This is included in the Pauli_strings. '''

    hmat_comp = csr_matrix((2**(nqubits+1),2**(nqubits+1)), dtype=np.complex128)
    k = 0
    for string in hamil_pauli_strings:
        k +=1
        circ = Circuit(nqubits+1)
        for i in range(len(string[0])):
            if string[0][i] == Pauli.Z:
                circ.Z(string[2][i])
            if string[0][i] == Pauli.X:
                circ.X(string[2][i])
            if string[0][i] == Pauli.Y:
                circ.Y(string[2][i])
        mat = csr_matrix(string[1] * circ.get_unitary())
        hmat_comp += mat
    circ = Circuit(nqubits+1)
    circ.Z(0)
    zmat_comp = csr_matrix(circ.get_unitary())
    zmat = zmat_comp.real
    hmat = hmat_comp.real
    save_npz(filename+'hmat', hmat)   
    save_npz(filename+'zmat', zmat)   
    return hmat, zmat

def get_hamil_matrix(hamil_pauli_strings:dict, nqubits:int):
    ''' Get matrices for each pauli string in a Hamiltonian.'''
    hmats = {}
    for string in hamil_pauli_strings:
        circ = Circuit(nqubits)
        for i in range(1, len(string[0])):
            if string[0][index] == Pauli.X:
                measurement_circ.X(string[1][index]-1)
            if string[0][index] == Pauli.Y:
                measurement_circ.Y(string[1][index]-1)
            if string[0][index] == Pauli.Z:
                measurement_circ.Z(string[1][index]-1)
        mat = csr_matrix(string[1] * circ.get_unitary())
        hmats[string] = mat.real
    return hmats


def get_hamil_group_matrices(ps_g:dict, nqubits:int, filename:str):
    ''' Function to generate Hamiltonian matrices for each commuting Pauli group
        for a Hadamard-test-type measurement circuit with 1 ancilla. To measure 
        <Psi_k|O|Psi_l> the Paulis corresponding to O must be applied to the 
        system qubits and a Z gate to the ancilla. This is included in the Pauli_strings. '''
    mats = {}
    k = 0
    for pauli_gr in ps_g:
        k +=1
        measurement_circ = Circuit(nqubits+1)
        for index in range(len(pauli_gr[0])):
            if pauli_gr[0][index] == Pauli.X:
                measurement_circ.H(pauli_gr[1][index])
            if pauli_gr[0][index] == Pauli.Y:
                measurement_circ.V(pauli_gr[1][index])
        mats[pauli_gr] = csr_matrix(measurement_circ.get_unitary())
    f  =open(filename, 'wb')
    pickle.dump(mats, f)
    f.close()
    return mats
