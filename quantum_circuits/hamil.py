from openfermion import MolecularData, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf
import utils


def get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge):

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1)
    qubit_h = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    return qubit_h

def get_hamil_pauli_strings(qubit_h):

    return utils.pauli_strings_from_fermion_operators_measurements([qubit_h])
