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
from openmmml import MLPotential
from physicsml.plugins.openmm.physicsml_potential import (
    MLPotential as PhysicsMLPotential,
)


class PotentialFactory:

    def __init__(self) -> None:
        pass

    @staticmethod
    def initialize_potential(
        params: Dict[str, Union[str, float, int]]
    ) -> Union["MLPotential", "PhysicsMLPotential"]:
        """
        Initializes a potential based on the provided parameters.

        Parameters
        ----------
        params : Dict[str, Union[str, float, int]]
            A dictionary containing the parameters for the potential. The required keys are:
            - "provider": The provider of the potential, either "openmm-ml" or "physics-ml".
            - "model_name": The name of the model.
            - "precision": The precision of the model (only for "physics-ml" provider).
            - "position_scaling": The scaling factor for the position (only for "physics-ml" provider).
            - "output_scaling": The scaling factor for the output (only for "physics-ml" provider).
            - "model_path": The path to the model file (only for "physics-ml" provider).

        Returns
        -------
        MLPotential
            An instance of the appropriate potential class based on the provided parameters.
        """

        log.info(
            f"Initialize {params['model_name']} potential from {params['provider']}"
        )
        kwargs = {}
        if params["provider"] == "openmm-ml":
            kwargs["name"] = params["model_name"].lower()
        elif params["provider"] == "physics-ml":

            kwargs["name"] = "physicsml_model"  # that key word needs to be present
            kwargs["precision"] = str(params["precision"]) # NOTE: precision has to be passed as str 
            kwargs["position_scaling"] = float(params["position_scaling"])
            kwargs["output_scaling"] = float(params["output_scaling"])
            kwargs["model_path"] = params.get("model_path", None)
            kwargs["repo_url"] = params.get("repo_url", None)
            kwargs["rev"] = params.get("rev", None)
            kwargs["device"] = params.get("device", None)
            kwargs["model_path_in_repo"] = params.get("model_path_in_repo", None)
        else:
            raise RuntimeError(f"Unsupported potential type: {params}")

        return MLPotential(**kwargs)
