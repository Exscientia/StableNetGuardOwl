import logging
from typing import List, Union

from loguru import logger as log
from openff.units.openmm import to_openmm
from openmm import LangevinIntegrator, Platform, unit
from openmm.app import Simulation
from openmmtools.testsystems import (AlanineDipeptideExplicit,
                                     AlanineDipeptideVacuum, SrcExplicit,
                                     TestSystem, WaterBox)
from openmmtools.utils import get_fastest_platform

from .constants import collision_rate, stepsize, temperature
from .setup import create_system_from_mol, generate_molecule


class SmallMoleculeVacuum:
    def __init__(self, name: str, smiles: str):
        """Initializes a small molecule testsystem

        Args:
            zink_id (str)
            smiles (str)
        """

        self.testsystem_name = name
        self.smiles = smiles

        mol = generate_molecule(self.smiles)

        self.system, topology = create_system_from_mol(mol)
        self.topology = topology.to_openmm()

        # set position
        self.positions = to_openmm(mol.conformers[0])
        self.mol = mol


class HipenSystemVacuum:
    def __init__(self, zink_id: str, smiles: str):
        """Initializes a HiPen testsystem

        Args:
            zink_id (str)
            smiles (str)
        """

        self.zink_id = zink_id
        self.testsystem_name = zink_id
        self.smiles = smiles

        mol = generate_molecule(self.smiles)

        self.system, topology = create_system_from_mol(mol)
        self.topology = topology.to_openmm()

        # set position
        self.positions = to_openmm(mol.conformers[0])


class HipenTestsystemFactory:
    def __init__(self) -> None:
        """Factory that returns HipenSystemVacuum"""
        self.hipen_systems = hipen_systems
        self.name = "hipen_testsystem"

    def generate_testsystems(self, name: str) -> HipenSystemVacuum:
        return HipenSystemVacuum(name, hipen_systems[name])


class SmallMoleculeTestsystemFactory:
    def __init__(self) -> None:
        """Factory that returns SmallMoleculeTestsystems"""
        self._mols = {
            "ethanol": "OCC",
            "methanol": "OC",
            "propanol": "OCC",
            "methane": "C",
        }

    def generate_testsystems(self, name: str) -> SmallMoleculeVacuum:
        if name not in list(self._mols.keys()):
            raise RuntimeError(
                f"Molecule is not in the list of available systems: {self._mols.keys()}"
            )

        return SmallMoleculeVacuum(name, self._mols[name])


class WaterboxTestsystemFactory:
    def __init__(self) -> None:
        """Factory that returns WaterBox testsystems with different edge lengths (10,20 of 30 Angstrom)

        Args:
            nr_of_systems (int, optional): defines the number of WaterBox testsystems that can be generated.
            Defaults to -1.
        """
        self.name = "waterbox_testsystem"

    def _run_sim(self, waterbox: WaterBox) -> WaterBox:
        integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
        platform = get_fastest_platform()

        sim = Simulation(
            waterbox.topology,
            waterbox.system,
            integrator,
            platform=platform,
        )
        sim.context.setPositions(waterbox.positions)
        sim.step(50_000)
        state = sim.context.getState(getPositions=True)
        waterbox.positions = (
            state.getPositions()
        )  # pylint: disable=unexpected-keyword-arg
        return waterbox

    def generate_testsystems(self, edge_length: unit.Quantity) -> WaterBox:
        waterbox = WaterBox(
            edge_length, cutoff=((edge_length / 2) - unit.Quantity(0.5, unit.angstrom))
        )
        print("Start equilibration ...")
        waterbox = self._run_sim(waterbox)
        print("Stop equilibration ...")
        return waterbox


class AlaninDipeptideTestsystemFactory:
    def __init__(self) -> None:
        """Factory that returns AlaninDipteptideTestsystem in vacuum and/or in solution"""
        self.name = "alanin_dipeptide_testsystem"

    def generate_testsystems(self, env: str) -> TestSystem:
        if env == "vacuum":
            return AlanineDipeptideVacuum(constraints=None)
        elif env == "solvent":
            return AlanineDipeptideExplicit(constraints=None)
        else:
            raise NotImplementedError("Only solvent and vacuum implemented")


class ProteinTestsystemFactory:
    """Factory that returns a protein testsystem (currently only the src protein)"""

    def __init__(self) -> None:
        self.protein_testsystems = {"src": SrcExplicit}
        self.name = "protein_testsystem"

    def generate_testsystems(self, name: str) -> TestSystem:
        if name == "src":
            return self.protein_testsystems["src"]


hipen_systems = {
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
