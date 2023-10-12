import csv
import os
import sys
from abc import ABC
from dataclasses import dataclass, field
from typing import List, TextIO, Tuple, Union, Optional

import mdtraj as md
import numpy as np
import openmm
from loguru import logger as log
from openmm import Platform, State, System, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter, Topology
from openmmtools.testsystems import TestSystem

from .simulation import SimulationFactory


def initialize_ml_system(nnp: str, topology: Topology, implementation: str) -> System:
    from openmmml import MLPotential

    from guardowl.simulation import SystemFactory

    nnp_instance = MLPotential(nnp)
    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance, topology, implementation=implementation
    )
    return system


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
    temperature: Union[int, List[int]]
    env: str
    simulated_annealing: bool
    state_data_reporter: StateDataReporter
    device_index: int = 0
    ensemble: Optional[str] = None


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
        assert parameters.env in ["vacuum", "solution"]
        ensemble = parameters.ensemble

        if ensemble:
            ensemble = parameters.ensemble.lower()
            assert ensemble in self.implemented_ensembles

        log.debug(
            f""" 
------------------------------------------------------------------------------------
Stability test parameters:
params.protocol_length: {parameters.protocol_length}
params.temperature: {parameters.temperature}
params.ensemble: {parameters.ensemble}
params.env: {parameters.env}
params.simulated_annealing: {parameters.simulated_annealing}
params.platform: {parameters.platform.getName()}
params.device_index: {parameters.device_index}
params.output_folder: {parameters.output_folder}
params.log_file_name: {parameters.log_file_name}
------------------------------------------------------------------------------------
            """
        )

        system = parameters.system

        qsim = SimulationFactory.create_simulation(
            system,
            parameters.testsystem.topology,
            platform=parameters.platform,
            temperature=temperature,
            env=parameters.env,
            device_index=parameters.device_index,
            ensemble=ensemble,
        )

        os.makedirs(parameters.output_folder, exist_ok=True)
        output_file_name = f"{parameters.output_folder}/{parameters.log_file_name}"

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
            env="vacuum",
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
        if not isinstance(parms.temperature, list):
            raise RuntimeError(
                "You need to provide mutliple temperatures to run the MultiTemperatureProtocol."
            )
        for temperature in parms.temperature:
            parms.log_file_name = f"{log_file_name_}_{temperature}"

            self._run_simulation(
                parms,
                temperature * unit.kelvin,
            )


def run_hipen_protocol(
    hipen_idx: int,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
    reporter: StateDataReporter,
    platform: openmm.Platform,
    output_folder: str,
    device_index: int = 0,
    nr_of_simulation_steps: int = 5_000_000,
):
    """
    Perform a stability test for a hipen molecule in vacuum
    at multiple temperatures with a nnp/implementation.
    :param hipen_idx: The index of the hipen molecule to simulate.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param nr_of_simulation_steps: The number of simulation steps to perform (default=5_000_000).
    """
    from guardowl.testsystems import HipenTestsystemFactory, hipen_systems

    name = list(hipen_systems.keys())[hipen_idx]

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing vacuum stability test for {name} from the hipen dataset in vacuum.
|  The simulation will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = HipenTestsystemFactory().generate_testsystems(name)
    system = initialize_ml_system(nnp, testsystem.topology, implementation)
    log_file_name = f"vacuum_{name}_{nnp}_{implementation}"
    if isinstance(temperature, int):
        stability_test = PropagationProtocol()
    else:
        stability_test = MultiTemperatureProtocol()

    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=temperature,
        ensemble="nvt",
        simulated_annealing=False,
        env="vacuum",
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
        device_index=device_index,
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def run_waterbox_protocol(
    edge_length: int,
    ensemble: str,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
    reporter: StateDataReporter,
    platform: openmm.Platform,
    output_folder: str,
    device_index: int = 0,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
):
    """
    Perform a stability test for a waterbox with a given edge size
    in PBC in an ensemble and with a nnp/implementation.
    :param edge_length: The edge length of the waterbox in Angstrom.
    :param ensemble: The ensemble to simulate in.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param annealing: Whether to perform simulated annealing (default=False).
    :param nr_of_simulation_steps: The number of simulation steps to perform (default=5_000_000).
    """
    from openmm import unit
    from guardowl.testsystems import WaterboxTestsystemFactory

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing waterbox stability test for a {edge_length} A waterbox in PBC.
|  The simulation will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = WaterboxTestsystemFactory().generate_testsystems(
        unit.Quantity(edge_length, unit.angstrom)
    )
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    log_file_name = f"waterbox_{edge_length}A_{nnp}_{implementation}_{ensemble}"
    log.info(f"Writing to {log_file_name}")

    stability_test = PropagationProtocol()

    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=temperature,
        ensemble=ensemble,
        simulated_annealing=annealing,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
        device_index=device_index,
        env="solution",
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def run_alanine_dipeptide_protocol(
    nnp: str,
    implementation: str,
    temperature: int,
    reporter: StateDataReporter,
    platform: openmm.Platform,
    output_folder: str,
    device_index: int = 0,
    ensemble: Optional[str] = None,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
    env: str = "vacuum"
):
    """
    Perform a stability test for an alanine dipeptide in water
    in PBC in an ensemble and with a nnp/implementation.
    :param env: The environment to simulate in (either vacuum or solvent).
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param ensemble: The ensemble to simulate in (default='').
    :param nr_of_simulation_steps: The number of simulation steps to perform (default=5_000_000).
    """
    from guardowl.testsystems import AlaninDipeptideTestsystemFactory

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing alanine dipeptide stability test in {env}.
|  The simulation will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = AlaninDipeptideTestsystemFactory().generate_testsystems(env=env)
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    if env == "vacuum":
        log_file_name = f"alanine_dipeptide_{env}_{nnp}_{implementation}"
    else:
        log_file_name = f"alanine_dipeptide_{env}_{nnp}_{implementation}_{ensemble}"

    log.info(f"Writing to {log_file_name}")

    stability_test = PropagationProtocol()

    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=temperature,
        ensemble=ensemble,
        simulated_annealing=annealing,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
        device_index=device_index,
        env=env,
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def run_DOF_scan(
    nnp: str,
    implementation: str,
    DOF_definition: dict,
    platform: openmm.Platform,
    output_folder: str,
    name: str = "ethanol",
):
    """
    Perform a scan on a selected DOF.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param DOF_definition: The DOF that is scanned. Allowed key values are 'bond', 'angle' and 'torsion',
    :param name: The name of the molecule to simulation (default='ethanol').
    values are a list of appropriate atom indices.
    """

    from guardowl.protocols import BondProfileProtocol, DOFTestParameters
    from guardowl.testsystems import SmallMoleculeTestsystemFactory

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing scan on a selected DOG for {name}.
|  The scan will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems(name)
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    log_file_name = f"vacuum_{testsystem.testsystem_name}_{nnp}_{implementation}"

    if DOF_definition["bond"]:
        stability_test = BondProfileProtocol()
        params = DOFTestParameters(
            system=system,
            platform=platform,
            testsystem=testsystem,
            output_folder=output_folder,
            log_file_name=log_file_name,
            bond=DOF_definition["bond"],
        )
        stability_test.perform_bond_scan(params)
    elif DOF_definition["angle"]:
        raise NotImplementedError
    elif DOF_definition["torsion"]:
        raise NotImplementedError
