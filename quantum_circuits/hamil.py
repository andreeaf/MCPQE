'''Hamiltonian manipulation module'''

from openfermion import MolecularData, jordan_wigner, get_fermion_operator, QubitOperator
from openfermionpyscf import run_pyscf
import utils


def get_qubit_molecular_hamiltonian(geometry: list[str, list[int]],
                                    basis: str,
                                    multiplicity: int,
                                    charge: int):
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
    qubit_h = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    print(molecule.fci_energy)
    return molecule.fci_energy, qubit_h

def get_hamil_pauli_strings(qubit_h: QubitOperator):

    ''' Function to generate a list of parametrised Pauli strings corresponding
        to a qubit operator.'''

    return utils.pauli_strings_from_fermion_operators_measurements([qubit_h])
