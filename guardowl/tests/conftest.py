from typing import Tuple

import pytest
from openmm import System

from guardowl.setup import generate_molecule_from_smiles
from guardowl.testsystems import hipen_systems
from openmm.app import PDBFile


@pytest.fixture(scope="session")
def single_hipen_system() -> PDBFile:
    """
    Generate a hipen system.

    Returns:
        A tuple containing the generated system, topology, and molecule.
    """
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    pdb = generate_molecule_from_smiles(smiles)
    return pdb


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    # Create a temporary directory for the session
    temp_dir = tmpdir_factory.mktemp("data")

    # Yield the temporary directory path to the tests
    yield temp_dir
