from dataclasses import dataclass, field
from typing import List, Optional, Union

from loguru import logger as log
from openmm import unit
from openmm.app import Topology
from openmmtools.testsystems import (
    AlanineDipeptideExplicit,
    AlanineDipeptideVacuum,
    WaterBox,
)
from openmmtools.utils import get_fastest_platform


class Testsystem:

    def __init__(self, topology: Topology, positions: unit.Quantity):
        self.topology = topology
        self.positions = positions


@dataclass
class LiquidOption:
    name: str
    nr_of_copies: int = -1
    edge_length: unit.Quantity = field(default_factory=lambda: 10 * unit.angstrom)


@dataclass
class SmallMoleculeVacuumOption:
    name: str = ""
    smiles: str = ""
    path: str = ""

    def __str__(self) -> str:
        if self.name:
            return self.name
        elif self.smiles:
            return self.smiles
        else:
            return self.path


@dataclass
class SolvatedSystemOption:
    name: str


class TestsystemFactory:

    _AVAILABLE_SYSTEM_FOR_PURE_LIQUIDS = {
        "butane": "CCCC",
        "cyclohexane": "C1CCCCC1",
        "ethane": "CC",
        "isobutane": "CC(C)C",
        "methanol": "CO",
        "propane": "CCC",
    }

    _HIPEN_SYSTEMS = {
        "ZINC00061095": r"CCOc1ccc2nc(/N=C\c3ccccc3O)sc2c1",  # hipen 1
        "ZINC00077329": r"Cn1cc(Cl)c(/C=N/O)n1",  # hipen 2
        "ZINC00079729": r"S=c1cc(-c2ccc(Cl)cc2)ss1",  # hipen 3
        "ZINC00086442": r"CN1C(=O)C/C(=N\O)N(C)C1=O",  # hipen 4
        "ZINC00087557": r"NNC(=O)[C@H]1C(c2ccccc2)[C@@H]1C(=O)NN",  # hipen 5
        "ZINC00095858": r"CCO/C(O)=N/S(=O)(=O)c1ccccc1Cl",  # hipen 6
        "ZINC00107550": r"C/C(=N\O)c1oc(C)nc1C",  # hipen 7
        "ZINC00107778": r"O/N=C/C1=C(Cl)c2cc(Cl)ccc2OC1",  # hipen 8
        "ZINC00123162": r"CC(=O)/C(=N/Nc1ccc(Cl)cc1)C(=O)c1ccccc1",  # hipen 9
        "ZINC00133435": r"c1ccc(-c2nc3ccccc3nc2-c2ccccn2)nc1",  # hipen 10
        "ZINC00138607": r"O=C(CC1=NO[C@H](c2ccccc2O)N1)N1CCCC1",  # hipen 11
        "ZINC00140610": r"Cc1cc(C)c2c(=O)[nH]sc2n1",  # hipen 12
        "ZINC00164361": r"CCON1C(=O)c2ccccc2C1=O",  # hipen 13
        "ZINC00167648": r"Cc1ccc(COn2c(-c3ccccc3)nc3ccccc3c2=O)cc1",  # hipen 14
        "ZINC00169358": r"CC1=Cn2c(=O)c3ccccc3c(=O)n2C1",  # hipen 15
        "ZINC01755198": r"CC(C)C(=O)NNC(=O)C(C)C",  # hipen 16
        "ZINC01867000": r"c1ccc(-c2ccccc2-c2ccccc2)cc1",  # hipen 17
        "ZINC03127671": r"O=C(CSCC(=O)Nc1ccccc1)NNC(=O)c1ccccc1",  # hipen 18
        "ZINC04344392": r"CCOC(=O)NNC(=O)NCCCc1ccc2ccc3cccc4ccc1c2c34",  # hipen 19
        "ZINC04363792": r"Clc1cc(Cl)cc(/N=c2\ssnc2-c2ccccc2Cl)c1",  # hipen 20
        "ZINC06568023": r"O=C(NNC(=O)c1ccccc1)c1ccccc1",  # hipen 21
        "ZINC33381936": r"O=S(=O)(O/N=C1/CCc2ccccc21)c1ccc(Cl)cc1",  # hipen 22
    }

    _STANDARD_TEST_SYSTEMS = {
        "ethanol": "CCO",
        "methanol": "CO",
        "methane": "C",
        "propane": "CCC",
        "butane": "CCCC",
        "pentane": "CCCCC",
        "hexane": "CCCCCC",
        "cylohexane": "C1CCCCC1",
        "isobutane": "CC(C)C",
        "isopentane": "CCC(C)C",
        "propanol": "CCCO",
        "acetylacetone": "CC(=O)CC(=O)C",
        "acetone": "CC(=O)C",
        "acetamide": "CC(=O)N",
        "acetonitrile": "CC#N",
        "aceticacid": "CC(=O)O",
        "acetaldehyde": "CC=O",
        "benzene": "c1ccccc1",
        "ala_dipeptide": "N[C@H](C(=O)O)C(N)C(=O)[C@@H](C)O",
    }

    def generate_testsystem(
        self,
        testsystem_option: Union[
            LiquidOption, SmallMoleculeVacuumOption, SolvatedSystemOption
        ],
    ) -> Testsystem:

        # pure liquid --- either water or organic liquid
        if isinstance(testsystem_option, LiquidOption):
            # organic liquid
            if (
                testsystem_option.name
                in TestsystemFactory._AVAILABLE_SYSTEM_FOR_PURE_LIQUIDS.keys()
            ):
                return self._generate_organic_liquid_testsystem(
                    testsystem_option.name, testsystem_option.nr_of_copies
                )
            # water
            elif testsystem_option.name == "water":
                return self._generate_waterbox_testsystem(testsystem_option.edge_length)
            else:
                raise NotImplementedError(
                    f"Only the following molecules are implemented: {TestsystemFactory._AVAILABLE_SYSTEM_FOR_PURE_LIQUIDS.keys()}"
                )
        elif isinstance(testsystem_option, SmallMoleculeVacuumOption):
            if testsystem_option.name == "ala_dipeptide":
                ala = AlanineDipeptideVacuum(constraints=None)
                return Testsystem(ala.topology, ala.positions)
            else:
                return self._generate_small_molecule_testsystem(testsystem_option)

        elif isinstance(testsystem_option, SolvatedSystemOption):
            if testsystem_option.name == "ala_dipeptide":
                ala = AlanineDipeptideExplicit(constraints=None)
                return Testsystem(ala.topology, ala.positions)

        else:
            raise RuntimeError("No valid input provided")

    def _generate_small_molecule_testsystem(
        self, testsystem_option: SmallMoleculeVacuumOption
    ) -> Testsystem:
        if testsystem_option.smiles:
            return _SmallMoleculeFactory().generate_testsystem_from_smiles(
                smiles=testsystem_option.smiles
            )
        elif testsystem_option.path:
            return _SmallMoleculeFactory().generate_testsystems_from_sdf(
                path=testsystem_option.path
            )
        elif testsystem_option.name:
            return _SmallMoleculeFactory().generate_testsystems_from_name(
                name=testsystem_option.name
            )
        else:
            raise RuntimeError("No valid input provided")

    def _generate_organic_liquid_testsystem(
        self, name: str, nr_of_copies: int
    ) -> Testsystem:
        import numpy as np
        from openff.interchange.components._packmol import UNIT_CUBE, pack_box
        from openff.toolkit import ForceField, Molecule
        from openff.units import unit as ofunit

        self.nr_of_copies = nr_of_copies

        # generate the system
        solvent = Molecule.from_smiles(
            TestsystemFactory._AVAILABLE_SYSTEM_FOR_PURE_LIQUIDS[name]
        )
        log.info(f"Generating pure liquid box for {name}")
        log.info("Packmol is running ...")

        n_atoms = solvent.n_atoms * nr_of_copies
        # 25 Angstrom ...1431 atoms (water)
        # 30 Angstrom ...3000 atoms (water)
        # 35 Angstrom ...4014 atoms (water)
        if n_atoms < 50:
            edge_length = 10
        else:
            edge_length = np.round(0.002 * n_atoms) + 15
        # NOTE: original regression line Y = 0.003813*X + 19.27
        log.debug(f"Calculated intial {edge_length} Angstrom for {n_atoms} atoms")
        success = False  # Repeat until sucess is True
        fail_counter = 0

        # NOTE: we are using openff and openMM units here, be careful if you change anything

        while not success:
            increase_packing = 0
            try:
                log.debug(
                    f"Trying to pack {nr_of_copies} copies of {name} in box with edge length {edge_length+increase_packing} ..."
                )
                log.debug(
                    f"Box vector: {(edge_length + increase_packing) * UNIT_CUBE* ofunit.nanometer,}"
                )
                topology = pack_box(
                    molecules=[solvent],
                    number_of_copies=[nr_of_copies],
                    box_vectors=(edge_length + increase_packing)
                    * UNIT_CUBE
                    * ofunit.nanometer,
                )
                success = True
            except Exception as e:
                fail_counter += 1
                log.error(f"Packmol failed with the following error: {e}")
                increase_packing += 1
                if fail_counter > 10:
                    raise RuntimeError(f"Packmol failed with the following error: {e}")

        log.debug("Packmol has finished sucessfully ...")

        positions = topology.get_positions().to(ofunit.nanometer).magnitude
        return Testsystem(topology.to_openmm(), positions * unit.nanometer)

    def _generate_waterbox_testsystem(self, edge_length: unit.Quantity) -> WaterBox:
        """Generate a WaterBox test system.

        Parameters
        ----------
        edge_length : unit.Quantity
            Edge length for the waterbox.

        Returns
        -------
        WaterBox
            Generated test system.
        """
        waterbox = WaterBox(
            edge_length, cutoff=((edge_length / 2) - unit.Quantity(0.5, unit.angstrom))
        )
        return Testsystem(waterbox.topology, waterbox.positions)


from rdkit import Chem


class _SmallMoleculeFactory:

    @staticmethod
    def generate_testsystems_from_mol(
        mol: Chem.Mol, name: Optional[str] = None
    ) -> Testsystem:
        """Generate a SmallMoleculeVacuum test system.

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to generate test system from.
        name : str
            Name of the test system to generate.

        Returns
        -------
        SmallMoleculeVacuum
            Generated test system.
        """
        from .setup import generate_pdbfile_from_mol

        pdb = generate_pdbfile_from_mol(mol)
        return Testsystem(pdb.topology, pdb.positions)

    def generate_testsystem_from_smiles(self, smiles: str) -> Testsystem:
        """Generate a SmallMoleculeVacuum test system.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.

        Returns
        -------
        SmallMoleculeVacuum
            Generated test system.
        """
        from .setup import generate_molecule_from_smiles

        mol = generate_molecule_from_smiles(smiles)
        return self.generate_testsystems_from_mol(mol)

    def generate_testsystems_from_name(self, name: str) -> Testsystem:
        """Generate a SmallMoleculeVacuum test system.

        Parameters
        ----------
        name : str
            Name of the test system to generate.

        Returns
        -------
        SmallMoleculeVacuum
            Generated test system.
        """
        if name == "ala":
            ala = AlanineDipeptideVacuum()
            return Testsystem(ala.topology, ala.positions)
        if name in list(TestsystemFactory._STANDARD_TEST_SYSTEMS.keys()):
            return self.generate_testsystem_from_smiles(
                TestsystemFactory._STANDARD_TEST_SYSTEMS[name]
            )
        elif name in list(TestsystemFactory._HIPEN_SYSTEMS.keys()):
            return self.generate_testsystem_from_smiles(
                TestsystemFactory._HIPEN_SYSTEMS[name]
            )
        else:
            raise RuntimeError(
                f"Molecule is not in the list of available systems: {TestsystemFactory._HIPEN_SYSTEMS.keys()} and {TestsystemFactory._STANDARD_TEST_SYSTEMS.keys()}"
            )

    def generate_testsystems_from_sdf(self, path: str) -> Testsystem:
        """Generate a SmallMoleculeVacuum test system.

        Parameters
        ----------
        path : str
            Path to the sdf file.

        Returns
        -------
        SmallMoleculeVacuum
            Generated test system.
        """
        from .setup import generate_molecule_from_sdf

        log.debug(f"Generating test system from {path}")
        mol = generate_molecule_from_sdf(path)
        return self.generate_testsystems_from_mol(mol)
