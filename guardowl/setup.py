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
    Generate an openFF Molecule instance from a SMILES string and generate conformers.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    nr_of_conformations : int, optional
        The number of conformers to generate for the molecule. Defaults to 10.

    Returns
    -------
    Molecule
        The generated openFF Molecule instance with conformers.

    """
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.generate_conformers(n_conformers=nr_of_conformations)
    return molecule


def create_system_from_mol(mol: Molecule) -> Tuple[System, Topology]:
    """
    Create an OpenMM System and Topology instance from an openFF Molecule.

    Parameters
    ----------
    mol : Molecule
        The openFF Molecule instance to convert into an OpenMM system.

    Returns
    -------
    Tuple[System, Topology]
        A tuple containing the generated OpenMM System and Topology instances.
    """
    assert mol.n_conformers > 0
    log.debug("Using openff ...")

    topology = mol.to_topology()
    system = forcefield.create_openmm_system(topology)
    return (system, topology.to_openmm())


def generate_molecule_from_sdf():
    pass
