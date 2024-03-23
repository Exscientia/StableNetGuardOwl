import pytest
from guardowl.testsystems import (
    PureLiquidTestsystemFactory,
    SmallMoleculeTestsystemFactory,
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
    factory = SmallMoleculeTestsystemFactory()
    sdf_file = f"{get_data_filename('tests/data/156613987')}/156613987.sdf"
    testsystem = factory.generate_testsystems_from_sdf(sdf_file)
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
    "molecule_name", PureLiquidTestsystemFactory._AVAILABLE_SYSTEM.keys()
)
@pytest.mark.parametrize("nr_of_copies", [100, 200])
def test_generate_pure_liquids(molecule_name, nr_of_copies) -> None:
    """ "Test if we can generate a pure liquid"""

    factory = PureLiquidTestsystemFactory()
    factory.generate_testsystems(
        name=molecule_name, nr_of_copies=nr_of_copies, nr_of_equilibration_steps=1
    )
