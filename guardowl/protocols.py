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


class DOFTest:
    def __init__(self) -> None:
        """
        Initializes a new instance of the StabilityTest class.
        """
        from .simulation import SimulationFactory

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


class StabilityTest:
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
        from .simulation import SimulationFactory

        self.potential_simulation_factory = SimulationFactory()
        self.implemented_ensembles = ["npt", "nvt", "nve"]

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__

    def _assert_input(self, parameters: StabilityTestParameters):
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

    @staticmethod
    def _run_simulation(
        parameters: StabilityTestParameters,
        qsim: Simulation,
    ) -> None:
        from openmm.app import DCDReporter, PDBFile

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

    @staticmethod
    def _setup_simulation(
        parameters: StabilityTestParameters,
        minimization_tolerance=1 * unit.kilojoule_per_mole / unit.angstrom,
        minimize: bool = True,
    ) -> None:
        """
        Runs a simulation for stability tests on molecular systems.

        This method runs a simulation for stability tests on molecular systems. It takes in a StabilityTestParameters object and a temperature as input parameters. It checks if the simulated annealing flag is set to True or False and if the ensemble is implemented in the protocol. It then creates a simulation object using the SimulationFactory class and sets the positions of the system. It minimizes the energy of the system and runs a simulated annealing molecular dynamics simulation if the simulated annealing flag is set to True. Finally, it writes out the simulation data to files.

        Parameters
        ----------
        parameters: StabilityTestParameters
            The parameters for the stability test.

        Returns
        -------
        None
        """
        from .simulation import SimulationFactory

        system = parameters.system

        qsim = SimulationFactory.create_simulation(
            system,
            parameters.testsystem.topology,
            platform=parameters.platform,
            temperature=parameters.temperature,
            env=parameters.env,
            device_index=parameters.device_index,
            ensemble=parameters.ensemble,
        )

        qsim.context.setPositions(parameters.testsystem.positions)

        if minimize:
            log.info("Minimizing energy")
            log.debug(f"{minimization_tolerance=}")
            qsim.minimizeEnergy(tolerance=minimization_tolerance)

        # check if simulated_annealing is an atrribute of parameters
        if (
            hasattr(parameters, "simulated_annealing")
            and parameters.simulated_annealing
        ):
            print("Running Simulated Annealing MD")
            # every 1000 steps raise the temperature by 5 K, ending at 325 K
            for temp in np.linspace(0, 300, 60):
                qsim.step(100)
                temp = unit.Quantity(temp, unit.kelvin)
                qsim.integrator.setTemperature(temp)
                if parameters.output_folderensemble == "npt":
                    barostat = parameters.system.getForce(barostate_force_id)
                    barostat.setDefaultTemperature(temp)

        return qsim

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
        from openmm.app import PDBFile
        from .simulation import SimulationFactory
        import mdtraj as md

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

        parms.log_file_name = f"{parms.log_file_name}_{parms.temperature}"

        self._assert_input(parms)
        qsim = self._setup_simulation(parms)
        self._run_simulation(parms, qsim)


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
        log.debug(
            f""" 
------------------------------------------------------------------------------------
Minimization test parameters:
params.convergence_criteria: {parameters.convergence_criteria}
params.env: {parameters.env}
params.platform: {parameters.platform.getName()}
params.device_index: {parameters.device_index}
params.output_folder: {parameters.output_folder}
params.log_file_name: {parameters.log_file_name}
------------------------------------------------------------------------------------
            """
        )

    def perform_stability_test(
        self, parms: MinimizationTestParameters, minimize: bool = True
    ) -> State:
        from openmm.app import PDBFile

        self._assert_input(parms)
        qsim = self._setup_simulation(
            parms, minimization_tolerance=parms.convergence_criteria, minimize=minimize
        )

        output_file_name = f"{parms.output_folder}/{parms.log_file_name}"
        state = qsim.context.getState(getPositions=True, getEnergy=True)
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
    Perform a stability test for a hipen molecule in vacuum
    at multiple temperatures with a nnp/implementation.
    :param hipen_idx: The index of the hipen molecule to simulate.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param nr_of_simulation_steps: The number of simulation steps to perform (default=5_000_000).
    """
    from guardowl.testsystems import SmallMoleculeTestsystemFactory, hipen_systems

    def _run_protocol(hipen_idx: int):
        name = list(hipen_systems.keys())[hipen_idx]

        print(
            f""" 
------------------------------------------------------------------------------------
|  Performing vacuum stability test for {name} from the hipen dataset in vacuum.
|  The simulation will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
            """
        )

        testsystem = SmallMoleculeTestsystemFactory().generate_testsystems_from_name(
            name
        )
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

    if isinstance(hipen_idx, int):
        _run_protocol(hipen_idx)
    else:
        for hipen_idx_ in hipen_idx:
            _run_protocol(hipen_idx_)


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
        edge_length * unit.angstrom, nr_of_equilibrium_steps
    )
    system = initialize_ml_system(nnp, testsystem.topology, implementation)

    log_file_name = (
        f"waterbox_{edge_length}A_{nnp}_{implementation}_{ensemble}_{temperature}K"
    )
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


def run_pure_liquid_protocol(
    molecule_name: Tuple[str, List[str]],
    nr_of_molecule: Tuple[int, List[int]],
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
    Perform a stability test for a pure liquid with a given number of molecules
    in PBC in an ensemble and with a nnp/implementation.
    :param molecule_name: The name of the solvent molecule (ethane, butane, propane, methanol, cyclohexane, isobutane).
    :param nr_of_molecule: The number of solvent molecules.
    :param ensemble: The ensemble to simulate in.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param annealing: Whether to perform simulated annealing (default=False).
    :param nr_of_simulation_steps: The number of simulation steps to perform (default=5_000_000).
    """
    from guardowl.testsystems import PureLiquidTestsystemFactory

    if isinstance(molecule_name, str):
        molecule_name_ = [molecule_name]
        nr_of_molecule_ = [nr_of_molecule]
    else:
        molecule_name_ = molecule_name * len(nr_of_molecule)
        nr_of_molecule_ = [
            element for element in nr_of_molecule for _ in range(len(molecule_name))
        ]

    for name, n_atoms in zip(molecule_name_, nr_of_molecule_):
        print(
            f""" 
    ------------------------------------------------------------------------------------
    |  Performing pure liquid stability test for {n_atoms} {name} in PBC at {temperature}.
    |  The simulation will use the {nnp} potential with the {implementation} implementation.
    ------------------------------------------------------------------------------------
            """
        )

        testsystem = PureLiquidTestsystemFactory().generate_testsystems(
            name=name,
            nr_of_copies=n_atoms,
            nr_of_equilibration_steps=nr_of_equilibration_steps,
        )
        system = initialize_ml_system(nnp, testsystem.topology, implementation)

        log_file_name = (
            f"pure_liquid_{name}_{n_atoms}_{nnp}_{implementation}_{ensemble}"
        )
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
    platform: Platform,
    output_folder: str,
    device_index: int = 0,
    ensemble: Optional[str] = None,
    annealing: bool = False,
    nr_of_simulation_steps: int = 5_000_000,
    env: str = "vacuum",
):
    """
    Perform a stability test for an alanine dipeptide in water
    in PBC in an ensemble and with a nnp/implementation.
    :param env: The environment to simulate in (either vacuum or solution).
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
    assert env in ["vacuum", "solution"], f"Invalid input: {env}"
    if env == "vacuum":
        log_file_name = f"alanine_dipeptide_{env}_{nnp}_{implementation}"
    else:
        log_file_name = f"alanine_dipeptide_{env}_{nnp}_{implementation}_{ensemble}"

    log_file_name = f"{log_file_name}_{temperature}K"
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
    platform: Platform,
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


def run_detect_minimum_test(
    nnp: str,
    implementation: str,
    platform: Platform,
    output_folder: str,
    percentage: int = 10,
    only_molecules_below_10_heavy_atoms: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Perform a minimization on a selected compound.
    :param nnp: The neural network potential to use.
    :param implementation: The implementation to use.
    :param name: The name of the molecule to simulation.
    """
    from guardowl.testsystems import SmallMoleculeTestsystemFactory
    import mdtraj as md
    from .utils import (
        extract_drugbank_tar_gz,
        _generate_file_list_for_minimization_test,
        _generate_input_for_minimization_test,
    )

    # extract drugbank tar.gz file
    extract_drugbank_tar_gz()

    # generate all relevenat input files
    files = _generate_file_list_for_minimization_test(shuffel=True)

    # calculate the number of molecules to test
    nr_of_molecules = files["total_number_of_systems"]
    nr_of_molecules_to_test = int(nr_of_molecules * (percentage / 100))

    score = {}
    counter = 0

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing minimization for {nr_of_molecules_to_test} molecules.
|  The scan will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
        """
    )

    for (minimized_file, minimized_position), (
        start_file,
        start_position,
    ) in _generate_input_for_minimization_test(files):
        log.info(f"Minimize test: {counter}/{nr_of_molecules_to_test}")

        from openff.toolkit.topology import Molecule

        # extract directory and name of minimized file
        working_dir = "".join(start_file.split("/")[-1])
        name = os.path.basename(working_dir.removesuffix(".xyz"))

        sdf_file = "".join(start_file.split(".")[0]) + ".sdf"
        mol = Molecule.from_file(sdf_file, allow_undefined_stereo=True)

        log.debug(f"Testing {name}")

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
        try:
            reference_testsystem = (
                SmallMoleculeTestsystemFactory().generate_testsystems_from_sdf(sdf_file)
            )
        except ValueError as e:
            log.info(f"Skipping {name} because of {e}")
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
