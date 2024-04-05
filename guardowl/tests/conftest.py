import pytest

from guardowl.setup import generate_molecule_from_smiles, generate_pdbfile_from_mol
from openmm.app import PDBFile
from guardowl.testsystems import TestsystemFactory, SmallMoleculeVacuumOption


@pytest.fixture(scope="session")
def single_hipen_system() -> PDBFile:
    """
    Generate a hipen system.

    Returns:
        A tuple containing the generated system, topology, and molecule.
    """
    name = list(TestsystemFactory._HIPEN_SYSTEMS.keys())[1]
    opt = SmallMoleculeVacuumOption(name=name)
    pdb = TestsystemFactory().generate_testsystem(opt)
    return pdb


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    # Create a temporary directory for the session
    temp_dir = tmpdir_factory.mktemp("data")

    # Yield the temporary directory path to the tests
    yield temp_dir
