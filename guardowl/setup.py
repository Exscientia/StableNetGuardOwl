from io import StringIO
from typing import Optional

from loguru import logger as log
from openmm.app import PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem


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


from typing import Dict, Union


class PotentialFactory:

    # potential:
    #   - physicsml-model:
    #       - name: "physicsml_model"
    #       - precision: 64
    #       - position_scaling: 10.0
    #       - output_scaling: 4.184 * 627
    #       - model_path: "path_to_model"

    #   - openmmml:
    #       - name: "ANI2x"

    def __init__(self) -> None:
        pass

    @staticmethod
    def initialize_potential(params: Dict[str, Union[str, float, int]]):

        log.info(
            f"Initialize {params['model_name']} potential from {params['provider']}"
        )

        if params["provider"] == "openmm-ml":
            from openmmml import MLPotential

            name = params["model_name"]
            return MLPotential(name.lower())
        elif params["provider"] == "physics-ml":
            from physicsml.plugins.openmm.physicsml_potential import (
                MLPotential as PhysicsMLPotential,
            )

            print(params)
            name = "physicsml_model"  # that key word needs to be present
            precision = params["precision"]
            position_scaling = params["position_scaling"]
            output_scaling = params["output_scaling"]
            model_path = params["model_path"]

            return PhysicsMLPotential(
                name,
                model_path=model_path,
                precision=str(precision),  #
                position_scaling=float(position_scaling),
                output_scaling=float(eval(output_scaling)),
            )
        else:
            raise RuntimeError(f"Unsupported potential type: {params}")
