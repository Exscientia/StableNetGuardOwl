from typing import Tuple
from loguru import logger as log

from openmm import System
from openmm.app import PDBFile

from rdkit import Chem


def generate_molecule_from_smiles(smiles: str) -> Chem.Mol:
    """
    Generates a rdkit molecule instance from a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    Returns
    -------
    Chem.Mol
        A rdkit molecule instance with generated conformer.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)

    return molecule


def generate_pdbfile_from_mol(molecule: Chem.Mol) -> PDBFile:
    import io
    from rdkit import Chem

    pdb_block = Chem.MolToPDBBlock(molecule)
    pdb_file = io.StringIO(pdb_block)
    return PDBFile(pdb_file)


def generate_molecule_from_sdf(path: str) -> Chem.Mol:
    """
    Generates a rdkit molecule instance from an SDF file.

    Parameters
    ----------
    path : str
        The file path to the SDF file.

    Returns
    -------
    Chem.Mol
        A rdkit molecule instance loaded from the SDF file.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    suppl = Chem.SDMolSupplier(path)
    mol = next(suppl)
    mol = Chem.AddHs(mol)
    return mol
