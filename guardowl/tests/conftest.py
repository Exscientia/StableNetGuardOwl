from typing import Tuple

import pytest
from openff.toolkit.topology import Topology, Molecule
from openmm import System

from guardowl.setup import create_system_from_mol, generate_molecule_from_smiles
from guardowl.testsystems import hipen_systems


@pytest.fixture(scope="session")
def generate_hipen_system() -> Tuple[System, Topology, Molecule]:
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    mol = generate_molecule_from_smiles(smiles)
    system, topology = create_system_from_mol(mol)
    return (system, topology, mol)
