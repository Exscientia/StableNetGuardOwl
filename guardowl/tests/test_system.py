import pytest
from guardowl.testsystems import (
    PureLiquidTestsystemFactory,
    SmallMoleculeTestsystemFactory,
)
from guardowl.utils import get_data_filename


def test_generate_small_molecule(tmp_dir) -> None:
    """Test if we can generate a small molecule"""
    factory = SmallMoleculeTestsystemFactory()
    sdf_file = f"{get_data_filename('tests/data/156613987')}/156613987.sdf"
    system = factory.generate_testsystems_from_sdf(sdf_file)
    from openmm.app import PDBFile

    print(system.topology)  # <openmm.app.topology.Topology object>
    PDBFile.writeFile(
        system.topology,
        system.positions,
        open(f"{tmp_dir}/tmp1.pdb", "w"),
    )

    import copy

    system_copy = copy.copy(system)
    print(system_copy.topology)

    top = system_copy.topology
    for atom in top.atoms():
        print(atom.index, atom.name, atom.residue.name)

    for bond in top.bonds():
        print(bond)

    PDBFile.writeFile(
        system.topology,
        system_copy.positions,
        open(f"{tmp_dir}/tmp2.pdb", "w"),
    )


def test_generate_molecule(single_hipen_system) -> None:
    """Test if we can generate a molecule instance"""
    system, top, mol = single_hipen_system
    assert mol.n_conformers >= 1


def test_generate_system_top_instances(single_hipen_system) -> None:
    """Test if we can generate a system/topology instance"""
    system, topology, mol = single_hipen_system
    assert system.getNumParticles() > 0
    indices = [atom.index for atom in topology.atoms()]
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
