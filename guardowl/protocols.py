import csv
import os
import sys
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
from loguru import logger as log
from openmm import State, System, unit, Platform
from openmm.app import Simulation, StateDataReporter, Topology

from .parameters import (
    StabilityTestParameters,
    MinimizationTestParameters,
    DOFTestParameters,
)
from .reporter import ContinuousProgressReporter
from .simulation import initialize_ml_system


class StabilityTest:
    """
    Abstract base class for stability tests on molecular systems using OpenMM.
    """

    implemented_ensembles = ["npt", "nvt", "nve"]

    def __init__(self) -> None:
        from .simulation import SimulationFactory

        self.potential_simulation_factory = SimulationFactory()

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__

    def _assert_input(
        self, parameters: Union[StabilityTestParameters, MinimizationTestParameters]
    ):
        assert parameters.simulated_annealing in [
            True,
            False,
        ], f"Invalid input: {parameters.simulated_annealing}"
        assert parameters.env in [
            "vacuum",
            "solution",
        ], f"Invalid input: {parameters.env}"
        ensemble = parameters.ensemble

        if ensemble:
            ensemble = parameters.ensemble.lower()
            assert ensemble in self.implemented_ensembles, f"{ensemble} not implemented"

        log.debug(f"Stability test parameters: {parameters}")

    @staticmethod
    def _run_simulation(
        parameters: StabilityTestParameters,
        sim: Simulation,
    ) -> None:
        from openmm.app import DCDReporter, PDBFile
        from pathlib import Path

        output_folder = Path(parameters.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file_name = f"{parameters.output_folder}/{parameters.log_file_name}"

        # Writing PDB file
        PDBFile.writeFile(
            parameters.testsystem.topology,
            parameters.testsystem.positions,
            open(f"{output_file_name}.pdb", "w"),
        )

        parameters.state_data_reporter._out = open(
            f"{output_file_name}.csv", "w"
        )  # NOTE: write to file
        if not parameters.state_data_reporter._step:
            log.info("Setting step to True")
            parameters.state_data_reporter._step = True

        sim.reporters.append(
            DCDReporter(
                f"{output_file_name}.dcd",
                parameters.state_data_reporter._reportInterval,
            )
        )  # NOTE: align write out frequency of state reporter with dcd reporter
        sim.reporters.append(parameters.state_data_reporter)
        sim.reporters.append(
            ContinuousProgressReporter(
                sys.stdout,
                reportInterval=100,
                total_steps=parameters.protocol_length,
            )
        )

        sim.step(parameters.protocol_length)

    @staticmethod
    def _setup_simulation(
        parameters: Union[StabilityTestParameters, MinimizationTestParameters],
        minimization_tolerance: unit.Quantity = 1
        * unit.kilojoule_per_mole
        / unit.angstrom,
        minimize: bool = True,
    ) -> Simulation:
        """
        Sets up and optionally minimizes a simulation based on the provided parameters,
        and runs simulated annealing if specified.

        Parameters
        ----------
        parameters : Union[StabilityTestParameters, MinimizationTestParameters]
            The parameters defining the simulation environment, system, and conditions under which the simulation will be run.
        minimization_tolerance : unit.Quantity, optional
            The energy tolerance to which the system will be minimized. Default is 1 kJ/mol/Å.
        minimize : bool, optional
            Flag to determine whether the system should be energy-minimized before simulation. Default is True.

        Returns
        -------
        Simulation
            The configured OpenMM Simulation object.

        """

        from .simulation import SimulationFactory

        sim = SimulationFactory.create_simulation(
            parameters.system,
            parameters.testsystem.topology,
            platform=parameters.platform,
            temperature=parameters.temperature,
            env=parameters.env,
            device_index=parameters.device_index,
            ensemble=parameters.ensemble,
        )

        # Set initial positions
        sim.context.setPositions(parameters.testsystem.positions)

        # Perform energy minimization if requested
        if minimize:
            log.info("Minimizing energy")
            sim.minimizeEnergy(tolerance=minimization_tolerance)
            log.info("Energy minimization complete.")

        # Execute simulated annealing if enabled
        if getattr(parameters, "simulated_annealing", False):
            log.info("Running Simulated Annealing MD...")
            # every 100 steps raise the temperature by 10 K, ending at simulation temperatue
            for temp in np.linspace(
                0, parameters.temperature.unit_in_quantity(unit.kelvin), 10
            ):
                sim.step(100)
                temp = unit.Quantity(temp, unit.kelvin)
                sim.integrator.setTemperature(temp)
                if parameters.output_folderensemble == "npt":
                    # FIXME
                    barostat = parameters.system.getForce(barostate_force_id)
                    barostat.setDefaultTemperature(temp)

        return sim

    def perform_stability_test(
        self, params: Union[StabilityTestParameters, MinimizationTestParameters]
    ) -> None:
        raise NotImplementedError()


from abc import ABC, abstractmethod


class DOFTest(ABC):
    def __init__(self) -> None:
        """
        Initializes the DOFTest class, setting up the required simulation factory for conducting tests.
        """
        from .simulation import SimulationFactory

        self.potential_simulation_factory = SimulationFactory()

    def setup_simulation(self, parameters: DOFTestParameters) -> Simulation:
        from openmm.app import PDBFile
        from .simulation import SimulationFactory

        sim = SimulationFactory.create_simulation(
            parameters.system,
            parameters.testsystem.topology,
            platform=parameters.platform,
            temperature=unit.Quantity(300, unit.kelvin),
            env="vacuum",
        )

        # write pdb file
        pdb_path = f"{parameters.output_folder}/{parameters.log_file_name}.pdb"
        with open(pdb_path, "w") as pdb_file:

            PDBFile.writeFile(
                parameters.testsystem.topology,
                parameters.testsystem.positions,
                pdb_path,
            )
        return sim

    def perform_scan(self, parameters: DOFTestParameters) -> None:
        """
        Conducts a bond length scan for a specified system and saves the results.

        Parameters
        ----------
        parameters : DOFTestParameters
            The parameters defining the bond scan, including the system, platform, and output details.
        """
        import mdtraj as md

        sim = self.setup_simulation(parameters)
        potential_energy, conformations, bond_length = self.perform_DOF_scan(
            sim, parameters
        )

        file_path = f"{parameters.output_folder}/{parameters.log_file_name}"
        md.Trajectory(conformations, parameters.testsystem.topology).save(
            f"{file_path}.dcd"
        )
        # write csv file generated from bond_length and potential_energy
        with open(f"{file_path}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["bond distance [A]", "potential energy [kJ/mol]"])
            writer.writerows(zip(bond_length, potential_energy))

    @abstractmethod
    def perform_DOF_scan(
        self,
        sim: Simulation,
        parameters: DOFTestParameters,
    ):
        pass


class BondProfileProtocol(DOFTest):
    def __init__(self) -> None:
        """
        Initializes the BondProfileProtocol class.
        """
        super().__init__()

    def set_bond_length(
        self, position: unit.Quantity, atom1: int, atom2: int, length: float
    ) -> unit.Quantity:
        """
        Adjusts the bond length between two specified atoms in a given set of positions.

        Parameters
        ----------
        position : unit.Quantity
            The current positions of all atoms in the system.
        atom1 : int
            The index of the first atom in the bond.
        atom2 : int
            The index of the second atom in the bond.
        length : float
            The desired bond length (in angstroms).

        Returns
        -------
        unit.Quantity
            The updated positions of all atoms in the system.
        """
        diff = (position[atom2] - position[atom1]).value_in_unit(unit.angstrom)
        normalized_diff = diff / np.linalg.norm(diff)
        new_positions = position.copy()
        new_positions[atom2] = position[atom1] + normalized_diff * unit.Quantity(
            length, unit.angstrom
        )
        return unit.Quantity(new_positions, unit.angstrom)

    def perform_DOF_scan(
        self,
        sim: Simulation,
        parameters: DOFTestParameters,
    ) -> Tuple[List[float], List[unit.Quantity], List[float]]:
        """
        Performs a scan of the bond length between two atoms in a given position.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        parameters : DOFTestParameters
            The parameters for the degree of freedom test.

        Returns
        -------
        Tuple[List[float], List[unit.Quantity], List[float]]
            Lists of potential energies, conformations, and bond lengths, respectively.
        """
        assert parameters.bond, "Bond parameters must be specified for a bond scan."
        bond_atom1, bond_atom2 = parameters.bond
        initial_pos = parameters.testsystem.positions
        conformations, potential_energy, bond_length = [], [], []

        for b in np.linspace(0, 8, 80):  # in angstrom
            new_pos = self.set_bond_length(
                initial_pos,
                bond_atom1,
                bond_atom2,
                b,
            )
            sim.context.setPositions(new_pos)
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()
            potential_energy.append(energy.value_in_unit(unit.kilojoule_per_mole))
            conformations.append(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
            bond_length.append(b)

        return (potential_energy, conformations * unit.nanometer, bond_length)


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

        parms.log_file_name = f"{parms.log_file_name}_{parms.temperature}"

        self._assert_input(parms)
        sim = self._setup_simulation(parms)
        self._run_simulation(parms, sim)


class MinimizationProtocol(StabilityTest):
    def __init__(self) -> None:
        """
        Initializes the MinimizationProtocol class by calling the constructor of its parent class and setting the ensemble.

        Returns:
        --------
        None
        """
        super().__init__()

    def _assert_input(self, parameters: MinimizationTestParameters):
        log.debug(f"Stability test parameters: {parameters}")

    def perform_stability_test(
        self, parms: MinimizationTestParameters, minimize: bool = True
    ) -> State:
        from openmm.app import PDBFile

        self._assert_input(parms)
        sim = self._setup_simulation(
            parms, minimization_tolerance=parms.convergence_criteria, minimize=minimize
        )

        output_file_name = f"{parms.output_folder}/{parms.log_file_name}"
        state = sim.context.getState(getPositions=True, getEnergy=True)
        PDBFile.writeFile(
            parms.testsystem.topology,
            state.getPositions(),
            open(f"{output_file_name}.pdb", "w"),
        )
        return state


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
        import dataclasses

        if not isinstance(parms.temperature, list):
            raise RuntimeError(
                "You need to provide mutliple temperatures to run the MultiTemperatureProtocol."
            )

        for temperature in parms.temperature:
            _parms = dataclasses.replace(parms)
            _parms.temperature = temperature
            _parms.log_file_name = f"{parms.log_file_name}_{temperature}K"
            log.info(f"Running simulation at temperature: {temperature} K")
            self._assert_input(_parms)

            qsim = self._setup_simulation(_parms)
            self._run_simulation(_parms, qsim)


def run_hipen_protocol(
    hipen_idx: Union[int, List[int]],
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
    reporter: StateDataReporter,
    platform: Platform,
    output_folder: str,
    device_index: int = 0,
    nr_of_simulation_steps: int = 5_000_000,
):
    """
    Executes stability tests for specified hipen molecules in vacuum using a neural network potential (NNP)
    with a specific implementation at multiple temperatures.

    Parameters
    ----------
    hipen_idx : Union[int, List[int]]
        The index or indices of the hipen molecule(s) to simulate.
    nnp : str
        The neural network potential to use for the simulation.
    implementation : str
        The specific implementation of the NNP.
    temperature : Union[int, List[int]]
        The temperature or list of temperatures at which to perform the simulations.
        Multiple temperatures trigger a multi-temperature protocol.
    reporter : StateDataReporter
        The OpenMM StateDataReporter for logging simulation progress.
    platform : Platform
        The OpenMM Platform on which to run the simulations.
    output_folder : str
        The directory path where output files will be saved.
    device_index : int, optional
        The index of the GPU device to use for the simulations, defaults to 0.
    nr_of_simulation_steps : int, optional
        The total number of simulation steps to perform, defaults to 5,000,000.

    """
    from guardowl.testsystems import SmallMoleculeTestsystemFactory, hipen_systems

    def _run_protocol(hipen_idx: int):
        name = list(hipen_systems.keys())[hipen_idx]

        log.info(
            f"Performing vacuum stability test for {name} using {nnp} with {implementation}."
        )

        testsystem = SmallMoleculeTestsystemFactory().generate_testsystems_from_name(
            name
        )
        system = initialize_ml_system(nnp, testsystem.topology, implementation)
        log_file_name = f"vacuum_{name}_{nnp}_{implementation}"

        # Select protocol based on whether temperature is a list or a single value
        stability_test = (
            MultiTemperatureProtocol()
            if isinstance(temperature, list)
            else PropagationProtocol()
        )

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
        log.info(f"\nSaving {params.log_file_name} files to {params.output_folder}")

        # Run protocol for each specified hipen index
        if isinstance(hipen_idx, int):
            _run_protocol(hipen_idx)
        else:
            for idx in hipen_idx:
                _run_protocol(idx)


def run_waterbox_protocol(
    edge_length: int,
    ensemble: str,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
    reporter: StateDataReporter,
    platform: Platform,
    output_folder: str,
    device_index: int = 0,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
    nr_of_equilibrium_steps: int = 50_000,
):
    """
    Performs a stability test on a waterbox system with specified edge length using a
    neural network potential (NNP) and implementation in a given ensemble at multiple temperatures.

    Parameters
    ----------
    edge_length : int
        The edge length of the waterbox in Angstroms.
    ensemble : str
        The ensemble to simulate (e.g., 'NVT', 'NPT').
    nnp : str
        The neural network potential to use.
    implementation : str
        The specific implementation of the NNP.
    temperature : Union[int, List[int]]
        The simulation temperature or list of temperatures for multi-temperature protocols.
    reporter : StateDataReporter
        The OpenMM StateDataReporter for logging simulation progress.
    platform : Platform
        The OpenMM Platform on which to run the simulation.
    output_folder : str
        Directory where output files will be saved.
    device_index : int, optional
        The index of the GPU device to use, defaults to 0.
    annealing : bool, optional
        Whether to perform simulated annealing, defaults to False.
    nr_of_simulation_steps : int, optional
        Total number of simulation steps, defaults to 5,000,000.
    nr_of_equilibrium_steps : int, optional
        Number of equilibrium steps before the stability test, defaults to 50,000.

    """
    log.info(
        f"Initiating waterbox stability test: {edge_length}A edge, {nnp} potential, {implementation} implementation, {ensemble} ensemble."
    )
    from openmm import unit
    from guardowl.testsystems import WaterboxTestsystemFactory

    testsystem = WaterboxTestsystemFactory().generate_testsystems(
        edge_length * unit.angstrom, nr_of_equilibrium_steps
    )
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    log_file_name = f"waterbox_{edge_length}A_{nnp}_{implementation}_{ensemble}"
    if isinstance(temperature, list):
        log_file_name += f"_multi-temp"
    else:
        log_file_name += f"_{temperature}K"

    log.info(f"Simulation output will be written to {log_file_name}")

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
    log.info(f"Simulation files saved to {output_folder}")


def run_pure_liquid_protocol(
    molecule_name: Union[str, List[str]],
    nr_of_molecule: Union[int, List[int]],
    ensemble: str,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
    reporter: StateDataReporter,
    platform: Platform,
    output_folder: str,
    device_index: int = 0,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
    nr_of_equilibration_steps: int = 50_000,
):
    """
    Executes stability tests for specified pure liquid systems, each containing a defined number of molecules, using a neural network potential (NNP) with a specified implementation at various temperatures.

    Parameters
    ----------
    molecule_name : Union[str, List[str]]
        The name or list of names of the solvent molecule(s) to simulate (e.g., 'ethane', 'butane').
    nr_of_molecule : Union[int, List[int]]
        The number or list of numbers of solvent molecules for each solvent type to simulate.
    ensemble : str
        The ensemble to simulate (e.g., 'NVT', 'NPT').
    nnp : str
        The neural network potential to use.
    implementation : str
        The specific implementation of the NNP.
    temperature : Union[int, List[int]]
        The simulation temperature(s) for the stability test.
    reporter : StateDataReporter
        The OpenMM StateDataReporter for logging simulation progress.
    platform : Platform
        The OpenMM Platform on which to run the simulation.
    output_folder : str
        The directory where output files will be saved.
    device_index : int, optional
        The index of the GPU device to use, defaults to 0.
    annealing : bool, optional
        Whether to perform simulated annealing, defaults to False.
    nr_of_simulation_steps : int, optional
        The total number of simulation steps, defaults to 5,000,000.
    nr_of_equilibration_steps : int, optional
        The number of equilibration steps before the stability test, defaults to 50,000.

    """
    from guardowl.testsystems import PureLiquidTestsystemFactory

    # Ensure inputs are listified for uniform processing
    molecule_names = (
        [molecule_name] if isinstance(molecule_name, str) else molecule_name
    )
    nr_of_molecules = (
        [nr_of_molecule] if isinstance(nr_of_molecule, int) else nr_of_molecule
    )

    for name, n_atoms in zip(molecule_names, nr_of_molecules):
        log.info(
            f"Initiating pure liquid stability test for {n_atoms} molecules of {name} at {temperature}K."
        )

        testsystem = PureLiquidTestsystemFactory().generate_testsystems(
            name=name,
            nr_of_copies=n_atoms,
            nr_of_equilibration_steps=nr_of_equilibration_steps,
        )
        system = initialize_ml_system(nnp, testsystem.topology, implementation)

        temperature_str = (
            f"{temperature}K" if isinstance(temperature, int) else "multi-temp"
        )
        log_file_name = f"pure_liquid_{name}_{n_atoms}_{nnp}_{implementation}_{ensemble}_{temperature_str}"

        log.info(f"Simulation output will be written to {log_file_name}")

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
        log.info(f"Simulation files saved to {output_folder}")


def run_alanine_dipeptide_protocol(
    nnp: str,
    implementation: str,
    temperature: int,
    reporter: StateDataReporter,
    platform: Platform,
    output_folder: str,
    device_index: int = 0,
    ensemble: Optional[str] = None,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
    env: str = "vacuum",
):
    """
    Executes a stability test for an alanine dipeptide system within specified environmental conditions using a neural network potential (NNP) and its implementation.

    Parameters
    ----------
    nnp : str
        The neural network potential to use for the simulation.
    implementation : str
        The specific implementation of the NNP.
    temperature : int
        The temperature at which to perform the simulation, in Kelvin.
    reporter : StateDataReporter
        The OpenMM StateDataReporter for logging simulation progress.
    platform : Platform
        The OpenMM Platform on which to run the simulation.
    output_folder : str
        The directory where output files will be saved.
    device_index : int, optional
        The index of the GPU device to use, defaults to 0.
    ensemble : Optional[str], optional
        The ensemble to simulate (e.g., 'NVT', 'NPT'), defaults to None.
    annealing : bool, optional
        Whether to perform simulated annealing, defaults to False.
    nr_of_simulation_steps : int, optional
        The total number of simulation steps, defaults to 5,000,000.
    env : str, optional
        The environment to simulate in ('vacuum' or 'solution'), defaults to 'vacuum'.

    """
    log.info(
        f"Initiating alanine dipeptide stability test in {env} using {nnp} potential with {implementation} implementation."
    )
    from guardowl.testsystems import AlaninDipeptideTestsystemFactory

    testsystem = AlaninDipeptideTestsystemFactory().generate_testsystems(env=env)
    system = initialize_ml_system(nnp, testsystem.topology, implementation)
    env_str = "vacuum" if env == "vacuum" else f"{env}_{ensemble}"
    log_file_name = f"alanine_dipeptide_{env_str}_{nnp}_{implementation}_{temperature}K"

    log.info(f"Simulation output will be written to {log_file_name}")

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
    log.info(f"Simulation files saved to {output_folder}")


def run_DOF_scan(
    nnp: str,
    implementation: str,
    DOF_definition: Dict[str, list],
    platform: Platform,
    output_folder: str,
    name: str = "ethanol",
):
    """
    Executes a scan over a specified degree of freedom (DOF) for a given molecule using a neural
    network potential (NNP) and its implementation.

    Parameters
    ----------
    nnp : str
        The neural network potential to use for the simulation.
    implementation : str
        The specific implementation of the NNP.
    DOF_definition : Dict[str, list]
        The degrees of freedom to scan. Supported keys are 'bond', 'angle', and 'torsion'. Each key maps to a list of atom indices defining the DOF.
    platform : Platform
        The OpenMM Platform on which to run the simulation.
    output_folder : str
        The directory where output files will be saved.
    name : str, optional
        The name of the molecule for simulation, defaults to 'ethanol'.

    """
    log.info(
        f"Initiating DOF scan for {name} using {nnp} with {implementation} implementation."
    )

    from guardowl.protocols import BondProfileProtocol, DOFTestParameters
    from guardowl.testsystems import SmallMoleculeTestsystemFactory

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems_from_name(name)
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    log_file_name = f"DOF_scan_{name}_{nnp}_{implementation}"

    if "bond" in DOF_definition:
        protocol = BondProfileProtocol()
        dof_type = "bond"
    elif "angle" in DOF_definition:
        raise NotImplementedError("Angle DOF scans are not yet implemented.")
    elif "torsion" in DOF_definition:
        raise NotImplementedError("Torsion DOF scans are not yet implemented.")
    else:
        raise ValueError(
            "Unsupported DOF type. Supported types are: 'bond', 'angle', 'torsion'."
        )

    params = DOFTestParameters(
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        **DOF_definition,
    )
    log.info(
        f"Performing {dof_type} scan with DOF definition: {DOF_definition[dof_type]}"
    )
    protocol.perform_scan(params)
    log.info(f"Scan results saved to {output_folder}")


def run_detect_minimum(
    nnp: str,
    implementation: str,
    platform: Platform,
    output_folder: str,
    percentage: int = 10,
    only_molecules_below_10_heavy_atoms: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Performs a minimization test on a subset of molecules from the DrugBank database, comparing the energy minimized conformations between DFT and a specified neural network potential (NNP).

    Parameters
    ----------
    nnp : str
        The neural network potential to use for the minimization test.
    implementation : str
        The implementation details of the neural network potential.
    platform : Platform
        The OpenMM Platform to perform simulations on.
    output_folder : str
        The directory where output files will be saved.
    percentage : int, optional
        The percentage of the total number of molecules to test, defaults to 10.
    only_molecules_below_10_heavy_atoms : bool, optional
        If True, only tests molecules with fewer than 10 heavy atoms, defaults to False.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary with molecule names as keys and tuples of RMSD and energy difference as values.
    """
    from guardowl.testsystems import SmallMoleculeTestsystemFactory
    import mdtraj as md
    from .utils import (
        extract_drugbank_tar_gz,
        _generate_file_list_for_minimization_test,
        _generate_input_for_minimization_test,
    )

    # Extract DrugBank tar.gz file and prepare input files
    extract_drugbank_tar_gz()
    files = _generate_file_list_for_minimization_test(shuffle=True)

    # calculate the number of molecules to test
    nr_of_molecules = files["total_number_of_systems"]
    nr_of_molecules_to_test = int(nr_of_molecules * (percentage / 100))

    score = {}
    counter = 0

    log.info(
        f"Performing minimization for {nr_of_molecules_to_test} molecules using {nnp} with {implementation}."
    )

    for (minimized_file, minimized_position), (
        start_file,
        start_position,
    ) in _generate_input_for_minimization_test(files):
        log.info(f"Minimization test: {counter}/{nr_of_molecules_to_test}")

        from openff.toolkit.topology import Molecule

        # Extract directory and name of the molecule file
        working_dir = "".join(start_file.split("/")[-1])
        name = os.path.basename(working_dir.removesuffix(".xyz"))

        sdf_file = "".join(start_file.split(".")[0]) + ".sdf"
        mol = Molecule.from_file(sdf_file, allow_undefined_stereo=True)

        # test if not implemented elements are in molecule, if yes skip
        def _contains_unknown_elements(mol: Molecule) -> bool:
            for atom in mol.atoms:
                if atom.atomic_number >= 15:
                    log.debug(f"Skipping {name} because it contains unknown elements")
                    return True
            return False

        # test if molecules has below 10 heavy atoms, if yes return True
        def _below_10_heavy_atoms(mol: Molecule) -> bool:
            heavy_atoms = 0
            for atom in mol.atoms:
                if atom.atomic_number != 1:
                    heavy_atoms += 1
            if heavy_atoms > 10:
                log.debug(
                    f"Skipping {name} because it has more than 10 heavy atoms: {heavy_atoms} heavy atoms"
                )
                return False
            log.debug(
                f"Using {name} because it has less than 10 heavy atoms: {heavy_atoms} heavy atoms"
            )
            return True

        # ANI-2x is trained on limited elements, if molecule contains unknown elements skip
        if _contains_unknown_elements(mol):
            continue

        if only_molecules_below_10_heavy_atoms:
            if not _below_10_heavy_atoms(mol):
                continue

        #########################################
        #########################################
        # initialize the system that has been minimized using DFT
        from openff.interchange.exceptions import UnassignedBondError

        try:
            reference_testsystem = (
                SmallMoleculeTestsystemFactory().generate_testsystems_from_sdf(sdf_file)
            )
        except (ValueError, UnassignedBondError) as e:
            log.warning(f"Skipping {name} because of {e}")
            continue
        # set the minimized positions
        reference_testsystem.positions = minimized_position
        system = initialize_ml_system(
            nnp, reference_testsystem.topology, implementation
        )
        log_file_name = f"ref_{name}_{nnp}_{implementation}"

        params = MinimizationTestParameters(
            platform=platform,
            system=system,
            testsystem=reference_testsystem,
            output_folder=output_folder,
            log_file_name=log_file_name,
        )

        state = MinimizationProtocol().perform_stability_test(params, minimize=False)
        reference_energy = state.getPotentialEnergy()

        reference_traj = md.Trajectory(
            reference_testsystem.positions, reference_testsystem.topology
        )
        #########################################
        #########################################
        # initialize the system that will be minimized using NNPs

        minimize_testsystem = reference_testsystem
        minimize_testsystem.positions = start_position

        system = initialize_ml_system(nnp, minimize_testsystem.topology, implementation)
        log_file_name = f"minimize_{name}_{nnp}_{implementation}"

        params = MinimizationTestParameters(
            platform=platform,
            system=system,
            testsystem=minimize_testsystem,
            output_folder=output_folder,
            log_file_name=log_file_name,
        )

        state = MinimizationProtocol().perform_stability_test(params, minimize=True)

        minimized_energy = state.getPotentialEnergy()
        minimize_testsystem.positions = state.getPositions(asNumpy=True)

        # calculate the energy error between the NNP minimized and the DFT minimized conformation
        d_energy = abs(reference_energy - minimized_energy)

        minimized_traj = md.Trajectory(
            minimize_testsystem.positions, minimize_testsystem.topology
        )

        # calculate the RMSD between the NNP minimized and the DFT minimized conformation
        _score_minimized = md.rmsd(minimized_traj, reference_traj)[0]

        log.debug(f"RMSD: {_score_minimized}; Energy error: {d_energy}")
        score[name] = (_score_minimized, d_energy._value)

        counter += 1
        if counter >= nr_of_molecules_to_test:
            break

    # print the results to stdout
    print(f"{'Name':<40} {'RMSD [A]':<20} {'Energy error [kJ/mol]':<20}")
    for name, (rmsd, energy_error) in score.items():
        print(f"{name:<40} {rmsd:<20} {energy_error:<20}")

    return score
