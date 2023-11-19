from typing import List, Union, Optional
from functools import lru_cache
from loguru import logger as log
from openff.units.openmm import to_openmm
from openmm import LangevinIntegrator, unit, MonteCarloBarostat
from openmm.app import Simulation
from openmmtools.testsystems import (
    AlanineDipeptideExplicit,
    AlanineDipeptideVacuum,
    SrcExplicit,
    TestSystem,
    WaterBox,
)
from openmmtools.utils import get_fastest_platform

from .constants import collision_rate, stepsize, temperature


class PureLiquidBoxTestSystem(TestSystem):

    """Pure liquid box."""

    def __init__(self, molecule_name: str, nr_of_copies: int):
        from openff.toolkit import ForceField, Molecule
        from openff.interchange.components._packmol import UNIT_CUBE, pack_box
        from openff.units import unit
        import numpy as np

        self.nr_of_copies = nr_of_copies
        sage = ForceField("openff-2.0.0.offxml")

        TestSystem.__init__(self)

        # generate the system
        solvent = Molecule.from_smiles(
            PureLiquidTestsystemFactory._AVAILABLE_SYSTEM[molecule_name]
        )
        log.info(f"Generating pure liquid box for {molecule_name}")
        log.info("Packmol is running ...")

        n_atoms = solvent.n_atoms * nr_of_copies
        # 25 Angstrom ...1431 atoms (water)
        # 30 Angstrom ...3000 atoms (water)
        # 35 Angstrom ...4014 atoms (water)
        if n_atoms < 50:
            edge_length = 10
        else:
            edge_length = np.round(0.002 * n_atoms) + 20
        # NOTE: original regression line Y = 0.003813*X + 19.27
        log.debug(f"Calculated intial {edge_length} Angstrom for {n_atoms} atoms")
        success = False  # Repeat until sucess is True
        while not success:
            increase_packing = 0
            fail_counter = 0
            try:
                log.debug(
                    f"Trying to pack {nr_of_copies} copies of {molecule_name} in box with edge length {edge_length+increase_packing} ..."
                )
                topology = pack_box(
                    molecules=[solvent],
                    number_of_copies=[nr_of_copies],
                    box_vectors=(edge_length + increase_packing)
                    * UNIT_CUBE
                    * unit.nanometer,
                )
                success = True
            except Exception as e:
                fail_counter += 1
                log.error(f"Packmol failed with the following error: {e}")
                increase_packing += 1
                if fail_counter > 10:
                    raise RuntimeError(
                        f"Packmol failed with the following error: {e}"
                    ) from e
        log.debug("Packmol has finished ...")

        sage = ForceField("openff-2.0.0.offxml")
        system = sage.create_openmm_system(topology)

        positions = topology.get_positions().to(unit.nanometer)
        self.system, self.positions, self.topology = (
            system,
            positions.magnitude,
            topology.to_openmm(),
        )


class BaseMoleculeTestSystem:
    """Base class for molecule test systems.

    This class encapsulates the common functionality for creating
    molecule-based test systems.
    """

    def __init__(
        self,
        name: str,
        smiles: Optional[str],
        sdf_file: Optional[str] = None,
        positions: Optional[List] = None,
    ):
        from .setup import (
            create_system_from_mol,
            generate_molecule_from_smiles,
            generate_molecule_from_sdf,
        )

        self.testsystem_name = name
        self.smiles = smiles

        if self.smiles:
            mol = generate_molecule_from_smiles(self.smiles)
        elif sdf_file:
            mol = generate_molecule_from_sdf(sdf_file, positions)
        else:
            raise RuntimeError("Either smiles or sdf_file must be provided")

        self.system, topology = create_system_from_mol(mol)
        self.topology = topology.to_openmm()
        self.positions = to_openmm(mol.conformers[0])
        self.mol = mol


class SmallMoleculeVacuumTestSystem(BaseMoleculeTestSystem):
    """Class for small molecule in vacuum test systems.

    Parameters
    ----------
    name : str
        Name of the test system.
    smiles : str
        SMILES string of the molecule.
    """

    pass  # All functionality is currently in the base class


class HipenSystemVacuum(BaseMoleculeTestSystem):
    """Class for HiPen molecule in vacuum test systems.

    Parameters
    ----------
    zink_id : str
        ZINC identifier for the molecule.
    smiles : str
        SMILES string of the molecule.
    """

    def __init__(self, zink_id: str, smiles: str):
        super().__init__(zink_id, smiles)
        self.zink_id = zink_id


class HipenTestsystemFactory:
    """Factory for generating HiPen test systems.

    This factory class provides methods to generate HiPen test systems.
    """

    def __init__(self) -> None:
        """Factory that returns HipenSystemVacuum"""
        self.hipen_systems = hipen_systems
        self.name = "hipen_testsystem"

    def generate_testsystems(self, name: str) -> HipenSystemVacuum:
        """Generate a HiPen test system.

        Parameters
        ----------
        name : str
            Name of the test system to generate.

        Returns
        -------
        HipenSystemVacuum
            Generated test system.
        """
        return HipenSystemVacuum(name, hipen_systems[name])


class SmallMoleculeTestsystemFactory:
    """Factory for generating SmallMoleculeVacuum test systems.

    This factory class provides methods to generate SmallMoleculeVacuum test systems.
    """

    def __init__(self) -> None:
        """Factory that returns SmallMoleculeTestsystems"""
        self._mols = {
            "ethanol": "OCC",
            "methanol": "OC",
            "propanol": "OCC",
            "methane": "C",
        }

    @lru_cache(maxsize=None)
    def generate_testsystems(self, name: str) -> SmallMoleculeVacuumTestSystem:
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
        if name not in list(self._mols.keys()):
            raise RuntimeError(
                f"Molecule is not in the list of available systems: {self._mols.keys()}"
            )

        return SmallMoleculeVacuumTestSystem(name, self._mols[name])


class LiquidTestsystemFactory:
    def _run_equilibration(
        self,
        testsystem: Union[WaterBox, PureLiquidBoxTestSystem],
        nr_of_steps: int = 50_000,
    ) -> Union[WaterBox, PureLiquidBoxTestSystem]:
        """Run a simulation on the liquid box.

        Parameters
        ----------
        Union[WaterBox, PureLiquid] : WaterBox
            system to simulate.

        Returns
        -------
        Union[WaterBox, PureLiquid]
            The system after simulation.
        """
        integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
        platform = get_fastest_platform()

        sim = Simulation(
            testsystem.topology,
            testsystem.system,
            integrator,
            platform=platform,
        )
        barostat = MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 10)

        sim.system.addForce(barostat)

        sim.context.setPositions(testsystem.positions)
        sim.step(nr_of_steps)
        state = sim.context.getState(getPositions=True)
        testsystem.positions = (
            state.getPositions()
        )  # pylint: disable=unexpected-keyword-arg
        return testsystem


class PureLiquidTestsystemFactory(LiquidTestsystemFactory):
    """Factory for generating pure liquid systems"""

    _AVAILABLE_SYSTEM = {
        "butane": "CCCC",
        "cyclohexane": "C1CCCCC1",
        "ethane": "CC",
        "isobutane": "CC(C)C",
        "methanol": "CO",
        "propane": "CCC",
    }

    def __init__(self) -> None:
        pass

    def generate_testsystems(
        self,
        name: str,
        nr_of_copies: int = 500,
        nr_of_equilibration_steps: int = 50_000,
    ) -> PureLiquidBoxTestSystem:
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
        assert (
            name in PureLiquidTestsystemFactory._AVAILABLE_SYSTEM.keys()
        ), f"Available systems are {PureLiquidTestsystemFactory._AVAILABLE_SYSTEM.keys()}"

        liquid_box = PureLiquidBoxTestSystem(name, nr_of_copies)
        print("Start equilibration ...")
        liquid_box = self._run_equilibration(liquid_box, nr_of_equilibration_steps)
        print("Stop equilibration ...")
        return liquid_box


class WaterboxTestsystemFactory(LiquidTestsystemFactory):
    """Factory for generating WaterBox test systems.

    This factory class provides methods to generate WaterBox test systems with different edge lengths.
    """

    def __init__(self) -> None:
        self.name = "waterbox_testsystem"

    def generate_testsystems(
        self, edge_length: unit.Quantity, nr_of_equilibrium_steps: int = 5_000
    ) -> WaterBox:
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
        print("Start equilibration ...")
        waterbox = self._run_equilibration(waterbox, nr_of_equilibrium_steps)
        print("Stop equilibration ...")
        return waterbox


class AlaninDipeptideTestsystemFactory:
    """Factory for generating alanine dipeptide test systems.

    This factory class provides methods to generate alanine dipeptide test systems in vacuum or solution.
    """

    def __init__(self) -> None:
        """Factory that returns AlaninDipteptideTestsystem in vacuum and/or in solution"""
        self.name = "alanin_dipeptide_testsystem"

    def generate_testsystems(self, env: str) -> TestSystem:
        """Generate an alanine dipeptide test system.

        Parameters
        ----------
        env : str
            Environment in which the system should be generated, either 'vacuum' or 'solvent'.

        Returns
        -------
        TestSystem
            Generated test system.

        Raises
        ------
        NotImplementedError
            If the provided environment is neither 'vacuum' nor 'solution'.
        """
        if env == "vacuum":
            return AlanineDipeptideVacuum(constraints=None)
        elif env == "solution":
            return AlanineDipeptideExplicit(constraints=None)
        else:
            raise NotImplementedError("Only solution and vacuum implemented")


class ProteinTestsystemFactory:
    """Factory for generating protein test systems.

    This factory class currently only provides methods to generate the SrcExplicit protein test system.
    """

    def __init__(self) -> None:
        self.protein_testsystems = {"src": SrcExplicit}
        self.name = "protein_testsystem"

    def generate_testsystems(self, name: str) -> TestSystem:
        """Generate a protein test system.

        Parameters
        ----------
        name : str
            Name of the protein test system to generate.

        Returns
        -------
        TestSystem
            Generated test system.

        Raises
        ------
        NotImplementedError
            If the provided name does not match any available test systems.
        """

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
