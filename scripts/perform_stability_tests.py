# ------------------ IMPORTS ------------------#
import yaml
import warnings
from pathlib import Path

import fire
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
        remove_constraints=False,
    )
    log_file_name = f"vacuum_{name}_{nnp}_{implementation}"
    stability_test = MultiTemperatureProtocol()

    reporter = create_state_data_reporter()

    params = StabilityTestParameters(
        protocol_length=nr_of_simulation_steps,
        temperature=unit.Quantity(300, unit.kelvin),
        ensemble="nvt",
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
    )

    log.info(
        f""" 
------------------------------------------------------------------------------------
Stability test parameters:
params.protocol_length: {params.protocol_length}
params.temperature: {params.temperature}
params.ensemble: {params.ensemble}
params.simulated_annealing: {params.simulated_annealing}
params.platform: {params.platform.getName()}
params.output_folder: {params.output_folder}
params.log_file_name: {params.log_file_name}
------------------------------------------------------------------------------------
          """
    )

    stability_test.perform_stability_test(params)
    print(f"\nSaving {params.log_file_name} files to {params.output_folder}")


def perform_waterbox_protocol(
    edge_length: int,
    ensemble: str,
    nnp: str,
    implementation: str,
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
        temperature=unit.Quantity(300, unit.kelvin),
        ensemble=ensemble,
        simulated_annealing=annealing,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
    )

    log.info(
        f""" 
------------------------------------------------------------------------------------
Stability test parameters:
params.protocol_length: {params.protocol_length}
params.temperature: {params.temperature}
params.ensemble: {params.ensemble}
params.simulated_annealing: {params.simulated_annealing}
params.platform: {params.platform.getName()}
params.output_folder: {params.output_folder}
params.log_file_name: {params.log_file_name}
------------------------------------------------------------------------------------
          """
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
        temperature=unit.Quantity(300, unit.kelvin),
        ensemble=ensemble.lower(),
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
    )

    log.info(
        f""" 
------------------------------------------------------------------------------------
Stability test parameters:
params.protocol_length: {params.protocol_length}
params.temperature: {params.temperature}
params.ensemble: {params.ensemble}
params.simulated_annealing: {params.simulated_annealing}
params.platform: {params.platform.getName()}
params.output_folder: {params.output_folder}
params.log_file_name: {params.log_file_name}
------------------------------------------------------------------------------------
          """
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

def read_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":

    config = read_config("test_config.yaml")
    
    hipen_protocol_config = config.get("hipen_protocol", {})
    perform_hipen_protocol(**hipen_protocol_config)

    waterbox_protocol_config = config.get("waterbox_protocol", {})
    perform_waterbox_protocol(**waterbox_protocol_config)

