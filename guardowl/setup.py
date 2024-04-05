from typing import Tuple, Optional
from loguru import logger as log

from openmm.app import PDBFile

from rdkit import Chem
from rdkit.Chem import AllChem
from io import StringIO


def generate_molecule_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Generates an RDKit molecule instance from a SMILES string with added hydrogens and a generated 3D conformer.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.

    Returns
    -------
    Optional[Chem.Mol]
        An RDKit molecule instance with a generated 3D conformer, or None if molecule generation fails.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        log.error(f"Failed to generate molecule from SMILES: {smiles}")
        return None

    molecule = Chem.AddHs(molecule)
    if AllChem.EmbedMolecule(molecule) == -1:
        log.error(f"Failed to generate 3D conformer for molecule: {smiles}")
        return None

    return molecule


def generate_pdbfile_from_mol(molecule: Chem.Mol) -> Optional[PDBFile]:
    """
    Generates a PDBFile object from an RDKit molecule instance.

    Parameters
    ----------
    molecule : Chem.Mol
        The RDKit molecule instance.

    Returns
    -------
    Optional[PDBFile]
        An OpenMM PDBFile object representing the molecule, or None if conversion fails.
    """
    try:
        pdb_block = Chem.MolToPDBBlock(molecule)
        pdb_file = StringIO(pdb_block)
        return PDBFile(pdb_file)
    except Exception as e:
        log.error(f"Error generating PDB file from molecule: {e}")
        return None


def generate_molecule_from_sdf(path: str) -> Optional[Chem.Mol]:
    """
    Generates an RDKit molecule instance from an SDF file.

    Parameters
    ----------
    path : str
        The file path to the SDF file.

    Returns
    -------
    Optional[Chem.Mol]
        An RDKit molecule instance loaded from the SDF file, or None if loading fails.
    """
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    for mol in suppl:
        if mol is not None:
            return mol

    log.error(f"Failed to load molecule from SDF file: {path}")
    return None


class PotentialFactory:
    
    def __init__(self) -> None:
        pass
    
    def initialize_potential(self, params):

        nnp_instance = MLPotential(nnp)
