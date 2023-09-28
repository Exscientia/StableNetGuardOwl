import csv
from loguru import logger as log
import os
import sys
from abc import ABC
from dataclasses import dataclass, field
from typing import List, TextIO, Tuple, Union

import mdtraj as md
import numpy as np
from openmm import MonteCarloBarostat, Platform, State, System, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter
from openmmtools.testsystems import TestSystem

from .simulation import SimulationFactory


# StateDataReporter with custom print function
class ContinuousProgressReporter(object):
    """A class for reporting the progress of a simulation continuously.

    Parameters
    ----------
    iostream : TextIO
        Output stream to write the progress report to.
    total_steps : int
        Total number of steps in the simulation.
    reportInterval : int
        Interval at which to report the progress.

    Attributes
    ----------
    _out : TextIO
        The output stream.
    _reportInterval : int
        The report interval.
    _total_steps : int
        Total number of steps.
    """

    def __init__(self, iostream: TextIO, total_steps: int, reportInterval: int):
        self._out = iostream
        self._reportInterval = reportInterval
        self._total_steps = total_steps

    def describeNextReport(self, simulation: Simulation) -> Tuple:
        """
        Returns information about the next report.

        Parameters
        ----------
        simulation : Simulation
            The simulation to report on.

        Returns
        -------
        Tuple
            A tuple containing information about the next report.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation: Simulation, state: State) -> None:
        """
        Reports the progress of the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation to report on.
        state : State
            The state of the simulation.
        """
        progress = 100.0 * simulation.currentStep / self._total_steps
        self._out.write(f"\rProgress: {progress:.2f}")
        self._out.flush()


@dataclass
class BaseParameters:
    system: System
    platform: Platform
    testsystem: TestSystem
    output_folder: str
    log_file_name: str


@dataclass
class StabilityTestParameters(BaseParameters):
    """Parameters for a stability test.

    Parameters are stored as attributes.

    Attributes
    ----------
    protocol_length : int
        Length of the protocol in time units.
    temperature : unit.Quantity
        Temperature of the simulation.
    ensemble : str
        Ensemble type ('NVT', 'NPT', etc.).
    simulated_annealing : bool
        Whether simulated annealing is to be used.
    system : System
        The OpenMM System object.
    platform : Platform
        The OpenMM Platform object.
    testsystem : TestSystem
        The test system for the simulation.
    output_folder : str
        Path to the output folder.
    log_file_name : str
        Name of the log file.
    state_data_reporter : StateDataReporter
        The OpenMM StateDataReporter object.
    """

    protocol_length: int
    temperature: unit.Quantity
    ensemble: str
    simulated_annealing: bool
    state_data_reporter: StateDataReporter


@dataclass
class DOFTestParameters(BaseParameters):
    """Parameters for a degree of freedom (DOF) test.

    In addition to attributes in StabilityTestParameters, extra attributes for DOF tests are included.

    Attributes
    ----------
    bond : List
        List of atom pairs to be considered as bonds.
    angle : List
        List of atom triplets to be considered as angles.
    torsion : List
        List of atom quartets to be considered as torsions.
    """

    bond: List = field(default_factory=lambda: [])
    angle: List = field(default_factory=lambda: [])
    torsion: List = field(default_factory=lambda: [])


class DOFTest(ABC):
    """Abstract base class for DOF tests.

    Attributes
    ----------
    potential_simulation_factory : SimulationFactory
        Factory to generate simulation instances.
    """

    def __init__(self) -> None:
        self.potential_simulation_factory = SimulationFactory()

    def _set_bond_length(
        self, position: unit.Quantity, atom1: int, atom2: int, length: float
    ) -> unit.Quantity:
        """
        Sets the length of a bond between two atoms.

        Parameters
        ----------
        position : unit.Quantity
            The positions of the atoms.
        atom1 : int
            The index of the first atom.
        atom2 : int
            The index of the second atom.
        length : float
            The desired length of the bond.

        Returns
        -------
        unit.Quantity
            The new positions of the atoms.
        """
        diff = (position[atom2] - position[atom1]).value_in_unit(unit.angstrom)
        normalized_diff = diff / np.linalg.norm(diff)
        new_positions = position.copy()
        new_positions[atom2] = position[atom1] + normalized_diff * unit.Quantity(
            length, unit.angstrom
        )
        log.debug(f"{position=}")
        log.debug(f"{new_positions=}")
        return unit.Quantity(new_positions, unit.angstrom)

    def _perform_bond_scan(
        self,
        qsim: Simulation,
        parameters: DOFTestParameters,
    ) -> Tuple[List, List, List]:
        """
        Performs a scan of the length of a bond between two atoms.

        Parameters
        ----------
        qsim : Simulation
            The simulation to perform the scan on.
        parameters : DOFTestParameters
            The parameters for the DOF test.

        Returns
        -------
        Tuple[List, List, List]
            A tuple containing the conformations, potential energies, and bond lengths.
        """
        assert parameters.bond is not None
        bond_atom1, bond_atom2 = parameters.bond
        initial_pos = parameters.testsystem.positions
        conformations, potential_energy, bond_length = [], [], []
        for b in np.linspace(0, 8, 100):  # in angstrom
            new_pos = self._set_bond_length(
                initial_pos,
                bond_atom1,
                bond_atom2,
                b,
            )
            qsim.context.setPositions(new_pos)
            energy = qsim.context.getState(getEnergy=True).getPotentialEnergy()
            potential_energy.append(energy.value_in_unit(unit.kilojoule_per_mole))
            conformations.append(new_pos.value_in_unit(unit.nano * unit.meter))
            bond_length.append(b)

    def describeNextReport(self, simulation: Simulation) -> Tuple:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation: Simulation, state: State) -> None:
        progress = 100.0 * simulation.currentStep / self._total_steps
        self._out.write(f"\rProgress: {progress:.2f}")
        self._out.flush()


@dataclass
class StabilityTestParameters:
    """
    A dataclass for storing parameters for a stability test.

    Attributes
    ----------
    protocol_length : int
        The length of the protocol to run.
    temperature : unit.Quantity
        The temperature to run the simulation at.
    ensemble : str
        The ensemble to simulate.
    simulated_annealing : bool
        Whether to use simulated annealing.
    system : System
        The system to simulate.
    platform : Platform
        The platform to run the simulation on.
    testsystem : TestSystem
        The test system to simulate.
    output_folder : str
        The path to the output folder.
    log_file_name : str
        The name of the log file.
    state_data_reporter : StateDataReporter
        The reporter to use for state data.
    """

    protocol_length: int
    temperature: Union[int, List[int]]
    ensemble: str
    simulated_annealing: bool
    system: System
    platform: Platform
    testsystem: TestSystem
    output_folder: str
    log_file_name: str
    state_data_reporter: StateDataReporter


@dataclass
class DOFTestParameters:
    """
    A dataclass for storing parameters for a degree of freedom (DOF) test.

    Attributes
    ----------
    protocol_length : int
        The length of the protocol to run.
    temperature : unit.Quantity
        The temperature to run the simulation at.
    ensemble : str
        The ensemble to simulate.
    simulated_annealing : bool
        Whether to use simulated annealing.
    system : System
        The system to simulate.
    platform : Platform
        The platform to run the simulation on.
    testsystem : TestSystem
        The test system to simulate.
    output_folder : str
        The path to the output folder.
    log_file_name : str
        The name of the log file.
    state_data_reporter : StateDataReporter
        The reporter to use for state data.
    """

    system: System
    platform: Platform
    testsystem: TestSystem
    output_folder: str
    log_file_name: str
    bond: List = field(default_factory=lambda: [])
    angle: List = field(default_factory=lambda: [])
    torsion: List = field(default_factory=lambda: [])


class DOFTest(ABC):
    def __init__(self) -> None:
        """
        Initializes a new instance of the StabilityTest class.
        """
        self.potential_simulation_factory = SimulationFactory()

    def _set_bond_length(
        self, position: unit.Quantity, atom1: int, atom2: int, length: float
    ) -> unit.Quantity:
        """
        Sets the bond length between two atoms in a given position.

        Parameters
        ----------
        position : unit.Quantity
            The initial position of the atoms.
        atom1 : int
            The index of the first atom.
        atom2 : int
            The index of the second atom.
        length : float
            The desired bond length.

        Returns
        -------
        unit.Quantity
            The new position of the atoms with the updated bond length.
        """
        diff = (position[atom2] - position[atom1]).value_in_unit(unit.angstrom)
        normalized_diff = diff / np.linalg.norm(diff)
        new_positions = position.copy()
        new_positions[atom2] = position[atom1] + normalized_diff * unit.Quantity(
            length, unit.angstrom
        )
        log.debug(f"{position=}")
        log.debug(f"{new_positions=}")
        return unit.Quantity(new_positions, unit.angstrom)

    def _perform_bond_scan(
        self,
        qsim: Simulation,
        parameters: DOFTestParameters,
    ) -> Tuple[List, List, List]:
        """
        Performs a scan of the bond length between two atoms in a given position.

        Parameters
        ----------
        qsim : Simulation
            The simulation object.
        parameters : DOFTestParameters
            The parameters for the degree of freedom test.

        Returns
        -------
        Tuple[List, List, List]
            A tuple containing the potential energy, conformations, and bond length.
        """
        assert parameters.bond is not None
        bond_atom1, bond_atom2 = parameters.bond
        initial_pos = parameters.testsystem.positions
        conformations, potential_energy, bond_length = [], [], []
        for b in np.linspace(0, 8, 100):  # in angstrom
            new_pos = self._set_bond_length(
                initial_pos,
                bond_atom1,
                bond_atom2,
                b,
            )
            qsim.context.setPositions(new_pos)
            energy = qsim.context.getState(getEnergy=True).getPotentialEnergy()
            potential_energy.append(energy.value_in_unit(unit.kilojoule_per_mole))
            conformations.append(new_pos.value_in_unit(unit.nano * unit.meter))
            bond_length.append(b)
            log.debug(f"{b=}, {energy=}")
        return (potential_energy, conformations, bond_length)  #


class StabilityTest(ABC):
    """
    Abstract base class for stability tests on molecular systems.
    """

    def __init__(self) -> None:
        """
        Initializes a BondProfileProtocol object.

        This method initializes a BondProfileProtocol object and sets the potential_simulation_factory attribute to a new SimulationFactory object.
        It also sets the implemented_ensembles attribute to a list of strings representing the ensembles that are implemented in the protocol.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.potential_simulation_factory = SimulationFactory()
        self.implemented_ensembles = ["npt", "nvt", "nve"]

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__

    def _run_simulation(
        self,
        parameters: StabilityTestParameters,
        temperature: unit.Quantity,
    ) -> None:
        """
        Runs a simulation for stability tests on molecular systems.

        This method runs a simulation for stability tests on molecular systems. It takes in a StabilityTestParameters object and a temperature as input parameters. It checks if the simulated annealing flag is set to True or False and if the ensemble is implemented in the protocol. It then creates a simulation object using the SimulationFactory class and sets the positions of the system. It minimizes the energy of the system and runs a simulated annealing molecular dynamics simulation if the simulated annealing flag is set to True. Finally, it writes out the simulation data to files.

        Parameters
        ----------
        parameters: StabilityTestParameters
            The parameters for the stability test.
        temperature: unit.Quantity
            The temperature of the simulation.

        Returns
        -------
        None
        """
        assert parameters.simulated_annealing in [True, False]
        ensemble = parameters.ensemble.lower()
        if ensemble:
            assert ensemble in self.implemented_ensembles

        log.info(
            f""" 
------------------------------------------------------------------------------------
Stability test parameters:
params.protocol_length: {parameters.protocol_length}
params.temperature: {parameters.temperature}
params.ensemble: {parameters.ensemble}
params.simulated_annealing: {parameters.simulated_annealing}
params.platform: {parameters.platform.getName()}
params.output_folder: {parameters.output_folder}
params.log_file_name: {parameters.log_file_name}
------------------------------------------------------------------------------------
            """
        )

        system = parameters.system

        if ensemble == "npt":  # for NpT add barostat
            barostate = MonteCarloBarostat(
                unit.Quantity(1, unit.atmosphere), temperature
            )
            barostate_force_id = system.addForce(barostate)

        if ensemble == "nve":  # for NVE change to integrator without thermostate
            qsim = SimulationFactory.create_nvt_simulation(
                system,
                parameters.testsystem.topology,
                platform=parameters.platform,
                temperature=temperature,
            )
        else:
            qsim = SimulationFactory.create_simulation(
                system,
                parameters.testsystem.topology,
                platform=parameters.platform,
                temperature=temperature,
            )

        os.makedirs(parameters.output_folder, exist_ok=True)
        output_file_name = f"{parameters.output_folder}/{parameters.log_file_name}"

        PDBFile.writeFile(
            parameters.testsystem.topology,
            parameters.testsystem.positions,
            open(f"{output_file_name}.pdb", "w"),
        )
        PDBFile.writeFile(
            parameters.testsystem.topology,
            parameters.testsystem.positions,
            open(f"{output_file_name}.pdb", "w"),
        )

        parameters.state_data_reporter._out = open(
            f"{output_file_name}.csv", "w"
        )  # NOTE: write to file
        if parameters.state_data_reporter._step is False:
            log.info("Setting step to True")
            parameters.state_data_reporter._step = True
        qsim.context.setPositions(parameters.testsystem.positions)

        qsim.minimizeEnergy()

        if parameters.simulated_annealing:
            print("Running Simulated Annealing MD")
            # every 1000 steps raise the temperature by 5 K, ending at 325 K
            for temp in np.linspace(0, 300, 60):
                qsim.step(100)
                temp = unit.Quantity(temp, unit.kelvin)
                qsim.integrator.setTemperature(temp)
                if ensemble == "npt":
                    barostat = parameters.system.getForce(barostate_force_id)
                    barostat.setDefaultTemperature(temp)

        qsim.reporters.append(
            DCDReporter(
                f"{output_file_name}.dcd",
                parameters.state_data_reporter._reportInterval,
            )
        )  # NOTE: align write out frequency of state reporter with dcd reporter
        qsim.reporters.append(parameters.state_data_reporter)
        qsim.reporters.append(
            ContinuousProgressReporter(
                sys.stdout,
                reportInterval=100,
                total_steps=parameters.protocol_length,
            )
        )

        qsim.step(parameters.protocol_length)

    def perform_stability_test(self, params: StabilityTestParameters) -> None:
        raise NotImplementedError()


class BondProfileProtocol(DOFTest):
    def __init__(self) -> None:
        """
        Initializes the StabilityTest class by calling the constructor of its parent class.
        """
        super().__init__()

    def perform_bond_scan(self, parameters: DOFTestParameters) -> None:
        """
        Performs a bond scan on the given system and test system using the given parameters.

        Parameters:
        -----------
        parameters : DOFTestParameters
            The parameters to use for the bond scan.

        Returns:
        --------
        None
        """
        qsim = SimulationFactory.create_simulation(
            parameters.system,
            parameters.testsystem.topology,
            platform=parameters.platform,
            temperature=unit.Quantity(300, unit.kelvin),
        )

        PDBFile.writeFile(
            parameters.testsystem.topology,
            parameters.testsystem.positions,
            open(f"{parameters.output_folder}/{parameters.log_file_name}.pdb", "w"),
        )

        (potential_energy, conformations, bond_length) = super()._perform_bond_scan(
            qsim, parameters
        )
        md.Trajectory(conformations, parameters.testsystem.topology).save(
            f"{parameters.output_folder}/{parameters.log_file_name}.dcd"
        )
        # write csv file generated from bond_length and potential_energy
        with open(
            f"{parameters.output_folder}/{parameters.log_file_name}.csv", "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["bond distance [A]", "potential energy [kJ/mol]"])
            writer.writerows(zip(bond_length, potential_energy))


class PropagationProtocol(StabilityTest):
    def __init__(self) -> None:
        """
        Initializes the PropagationProtocol class by calling the constructor of its parent class and setting the ensemble.

        Returns:
        --------
        None
        """
        super().__init__()

    def perform_stability_test(
        self,
        parms: StabilityTestParameters,
    ) -> None:
        assert isinstance(parms.temperature, int)

        self._run_simulation(parms, parms.temperature * unit.kelvin)


class MultiTemperatureProtocol(PropagationProtocol):
    def __init__(self) -> None:
        """
        Initializes the PropagationProtocol class by calling the constructor of its parent class and setting the temperature protocol.

        Returns:
        --------
        None
        """
        super().__init__()

    def perform_stability_test(self, parms: StabilityTestParameters) -> None:
        """
        Performs a stability test on the molecular system for each temperature in the temperature protocol by running a simulation.

        Parameters:
        -----------
        parameters: StabilityTestParameters
            The parameters for the stability test.

        Returns:
        --------
        None
        """
        log_file_name_ = parms.log_file_name
        for temperature in parms.temperature:
            parms.log_file_name = f"{log_file_name_}_{temperature}"

            self._run_simulation(
                parms,
                temperature * unit.kelvin,
            )
