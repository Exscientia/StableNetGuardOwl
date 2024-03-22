from typing import Tuple
from loguru import logger as log

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import System


forcefield = ForceField("openff_unconstrained-2.0.0.offxml")


def generate_molecule_from_smiles(
    smiles: str, nr_of_conformations: int = 10
) -> Molecule:
    """
    Generates an OpenFF Molecule instance from a SMILES string and generates conformers for it.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    nr_of_conformations : int, optional
        The number of conformers to generate for the molecule, by default 10.

    Returns
    -------
    Molecule
        An OpenFF Molecule instance with generated conformers.
    """
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.generate_conformers(n_conformers=nr_of_conformations)
    return molecule


def create_system_from_mol(mol: Molecule) -> Tuple[System, Topology]:
    """
    Creates an OpenMM System and Topology from an OpenFF Molecule.

    Parameters
    ----------
    mol : Molecule
        The OpenFF Molecule instance to convert into an OpenMM system.

    Returns
    -------
    Tuple[System, Topology]
        A tuple containing the generated OpenMM System and the corresponding Topology.
    """
    assert mol.n_conformers > 0, "Molecule must have at least one conformer."
    log.debug("Generating OpenMM system from OpenFF molecule.")

    topology = mol.to_topology()
    system = forcefield.create_openmm_system(topology)
    return (system, topology.to_openmm())


def generate_molecule_from_sdf(path: str) -> Molecule:
    """
    Generates an OpenFF Molecule instance from an SDF file.

    Parameters
    ----------
    path : str
        The file path to the SDF file.

    Returns
    -------
    Molecule
        An OpenFF Molecule instance loaded from the SDF file.
    """
    mol = Molecule.from_file(path, allow_undefined_stereo=True)
    log.info(f"Molecule loaded from SDF file: {path}")
    return mol
