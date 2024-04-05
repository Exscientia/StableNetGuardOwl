import pytest
from guardowl.testsystems import (
    TestsystemFactory,
    LiquidOption,
    SmallMoleculeVacuumOption,
)
from guardowl.utils import get_data_filename

from openmm.app import PDBFile


def test_different_ways_to_generate_top(tmp_dir) -> None:
    """Test if we can generate a molecule"""
    from guardowl.setup import (
        generate_pdbfile_from_mol,
        generate_molecule_from_smiles,
        generate_molecule_from_sdf,
    )
    from rdkit import Chem

    smiles = "CCO"
    mol = generate_molecule_from_smiles(smiles)
    assert isinstance(mol, Chem.Mol)

    sdf_file = f"{get_data_filename('tests/data/156613987')}/156613987.sdf"
    mol = generate_molecule_from_sdf(sdf_file)
    assert isinstance(mol, Chem.Mol)

    # test that we can derive topology correctly
    pdb = generate_pdbfile_from_mol(mol)
    top = pdb.topology
    positions = pdb.positions


def test_generate_small_molecule(tmp_dir) -> None:
    """Test if we can generate a small molecule"""
    factory = TestsystemFactory()
    sdf_file = f"{get_data_filename('tests/data/156613987')}/156613987.sdf"
    opt = SmallMoleculeVacuumOption(path=sdf_file)
    testsystem = factory.generate_testsystem(opt)
    assert testsystem is not None


def test_generate_molecule(single_hipen_system) -> None:
    """Test if we can generate a molecule instance"""
    pdb = single_hipen_system
    assert len(pdb.positions) >= 1


def test_generate_system_top_instances(single_hipen_system) -> None:
    """Test if we can generate a system/topology instance"""
    pdb = single_hipen_system
    assert pdb.topology.getNumAtoms() > 0
    indices = [atom.index for atom in pdb.topology.atoms()]
    assert len(indices) > 0


@pytest.mark.parametrize(
    "molecule_name", TestsystemFactory._AVAILABLE_SYSTEM_FOR_PURE_LIQUIDS.keys()
)
@pytest.mark.parametrize("nr_of_copies", [100, 200])
def test_generate_pure_liquids(molecule_name, nr_of_copies) -> None:
    """ "Test if we can generate a pure liquid"""

    opt = LiquidOption(name=molecule_name, nr_of_copies=nr_of_copies)
    factory = TestsystemFactory()
    factory.generate_testsystem(opt)
