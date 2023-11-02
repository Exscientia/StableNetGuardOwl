from guardowl.setup import create_system_from_mol, generate_molecule
import pytest
from guardowl.testsystems import PureLiquidTestsystemFactory, PureLiquidBoxTestSystem


def test_generate_molecule(generate_hipen_system) -> None:
    """Test if we can generate a molecule instance"""
    system, top, mol = generate_hipen_system
    assert mol.n_conformers >= 1


def test_generate_system_top_instances(generate_hipen_system) -> None:
    """Test if we can generate a system/topology instance"""
    system, topology, mol = generate_hipen_system
    topology = topology.to_openmm()

    indices = [atom.index for atom in topology.atoms()]
    assert len(indices) > 0


@pytest.mark.parametrize(
    "molecule_name", PureLiquidTestsystemFactory._AVAILABLE_SYSTEM.keys()
)
@pytest.mark.parametrize(
    "nr_of_copies",[100,200]
)
def test_generate_pure_liquids(molecule_name, nr_of_copies) -> None:
    """ "Test if we can generate a pure liquid"""

    factory = PureLiquidTestsystemFactory()
    factory.generate_testsystems(name=molecule_name, nr_of_copies=nr_of_copies, nr_of_equilibration_steps=1)
