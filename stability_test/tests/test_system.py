from stability_test.setup import create_system_from_mol, generate_molecule
from stability_test.testsystems import hipen_systems


def test_generate_molecule() -> None:
    """Test if we can generate a molecule instance"""
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    mol = generate_molecule(smiles)
    assert mol.n_conformers >= 1


def test_generate_system_top_instances() -> None:
    """Test if we can generate a system/topology instance"""
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    mol = generate_molecule(smiles)
    # define region that should be treated with the qml
    _, topology = create_system_from_mol(mol)
    topology = topology.to_openmm()

    indices = [atom.index for atom in topology.atoms()]
    assert len(indices) > 0
