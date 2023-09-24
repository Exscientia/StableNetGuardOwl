import logging
from typing import Tuple
from loguru import logger as log

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import System


forcefield = ForceField("openff_unconstrained-2.0.0.offxml")


def generate_molecule(smiles: str, nr_of_conformations: int = 10) -> Molecule:
    """
    generate a molecule using openff

    Args:
        smiles (str): SMILES string of the molecule
        nr_of_conformations (int, optional): Generates a number of conformation. Defaults to 10.

    Returns:
        Molecule: openff molecule instance
    """
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.generate_conformers(n_conformers=nr_of_conformations)
    # TODO: make sure that conformations are deterministic
    # NOTE: packmole, Modeller, PDBfixer to solvate
    return molecule


def create_system_from_mol(
    mol: Molecule, env: str = "vacuum"
) -> Tuple[System, Topology]:
    """Given a openFF Molecule instance an openMM system and topology instance is created

    Args:
        mol (Molecule): _description_
        env (str, optional): _description_. Defaults to 'vacuum'.

    Returns:
        _type_: _description_
    """
    assert mol.n_conformers > 0
    log.debug("Using openff ...")
    log.debug(f"Generating system in {env}")
    assert env in ("waterbox", "vacuum", "complex")
    ###################
    log.debug(f"{env=}")

    topology = mol.to_topology()
    system = forcefield.create_openmm_system(topology)
    return (system, topology)
