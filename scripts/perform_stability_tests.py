# ------------------ IMPORTS ------------------#
import yaml
import warnings
from pathlib import Path
import argparse
from typing import List, Union
from loguru import logger as log
from openmm import unit
from openmm.app import StateDataReporter
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform

from stability_test.protocolls import (
    BondProfileProtocol,
    DOFTestParameters,
    MultiTemperatureProtocol,
    PropagationProtocol,
    StabilityTestParameters,
)
from stability_test.simulation import SystemFactory
from stability_test.testsystems import (
    AlaninDipeptideTestsystemFactory,
    HipenTestsystemFactory,
    SmallMoleculeTestsystemFactory,
    WaterboxTestsystemFactory,
    hipen_systems,
)

from stability_test.utils import available_nnps_and_implementation

warnings.filterwarnings("ignore")
output_folder = "test_stability_protocol"

platform = get_fastest_platform()
log.info(f"Using platform {platform.getName()}")


def setup_logging_and_output():
    output_folder = "test_stability_protocol"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    return output_folder


def validate_input(nnp: str, implementation: str):
    if (nnp, implementation) not in available_nnps_and_implementation:
        error_message = f"Invalid nnp/implementation combination. Valid combinations are: {available_nnps_and_implementation}. Got {nnp}/{implementation}"
        log.error(error_message)
        raise RuntimeError(error_message)


def create_test_system(testsystem_factory, *args, **kwargs):
    return testsystem_factory().generate_testsystems(*args, **kwargs)


def initialize_ml_system(nnp, topology, implementation):
    nnp_instance = MLPotential(nnp)
    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance, topology, implementation=implementation
    )
    return system


def create_state_data_reporter():
    return StateDataReporter(
        file=None,
        reportInterval=500,
        step=True,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )


def perform_protocol(stability_test, params):
    log.info(f"Stability test parameters: {params}")
    stability_test.perform_stability_test(params)


def perform_hipen_protocol(
    hipen_idx: int,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
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
    validate_input(nnp, implementation)
    output_folder = setup_logging_and_output()

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
    nnp_instance = MLPotential(nnp)
    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )
    log_file_name = f"vacuum_{name}_{nnp}_{implementation}"
    if isinstance(temperature, int):
        stability_test = PropagationProtocol()
    else:
        stability_test = MultiTemperatureProtocol()

    reporter = create_state_data_reporter()

    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=temperature,
        ensemble="nvt",
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def perform_waterbox_protocol(
    edge_length: int,
    ensemble: str,
    nnp: str,
    implementation: str,
    temperature: Union[int, List[int]],
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
    validate_input(nnp, implementation)
    output_folder = setup_logging_and_output()

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
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    log_file_name = f"waterbox_{edge_length}A_{nnp}_{implementation}_{ensemble}"
    log.info(f"Writing to {log_file_name}")

    stability_test = PropagationProtocol()

    reporter = create_state_data_reporter()
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
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def perform_alanine_dipeptide_protocol(
    env: str,
    nnp: str,
    implementation: str,
    ensemble: str = "",
    nr_of_simulation_steps: int = 5_000_000,
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
    validate_input(nnp, implementation)
    output_folder = setup_logging_and_output()

    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing alanine dipeptide stability test with PBC.
|  The simulation will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = AlaninDipeptideTestsystemFactory().generate_testsystems(env=env)
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    log_file_name = f"alanine_dipeptide_{env}_{nnp}_{implementation}_{ensemble}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Writing to {log_file_name}")

    stability_test = PropagationProtocol()

    reporter = create_state_data_reporter()
    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=300,
        ensemble=ensemble.lower(),
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def perform_DOF_scan(
    nnp: str,
    implementation: str,
    DOF_definition: dict,
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
    validate_input(nnp, implementation)
    output_folder = setup_logging_and_output()
    print(
        f""" 
------------------------------------------------------------------------------------
|  Performing scan on a selected DOG for {name}.
|  The scan will use the {nnp} potential with the {implementation} implementation.
------------------------------------------------------------------------------------
          """
    )

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems(name)
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    log_file_name = f"vacuum_{testsystem.testsystem_name}_{nnp}_{implementation}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

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


def load_config(config_file_path):
    with open(config_file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def main():
    parser = argparse.ArgumentParser(
        description="Perform stability tests based on the YAML config file."
    )
    # Required argument for YAML configuration file
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        if config is None:
            log.warning("Error loading configuration.")
            raise RuntimeError("Error loading configuration.")
    else:
        log.warning("No configuration file provided.")
        raise RuntimeError("No configuration file provided.")

    # Do something with the config
    log.info(f"Loaded config: {config}")

    for test in config.get("tests", []):
        protocol = test.get("protocol")

        if protocol == "hipen_protocol":
            log.info("Performing hipen protocol")
            perform_hipen_protocol(**{k: test[k] for k in test if k != "protocol"})

        elif protocol == "waterbox_protocol":
            log.info("Performing waterbox protocol")
            perform_waterbox_protocol(**{k: test[k] for k in test if k != "protocol"})

        else:
            log.warning(f"Unknown protocol: {protocol}")


if __name__ == "__main__":
    main()
