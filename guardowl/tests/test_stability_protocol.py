from typing import Dict
from pathlib import Path

import pytest
from openmm import unit
from openmm.app import StateDataReporter
from exs.physicsml.plugins.openmm.mlpotential import MLPotential
from exs.physicsml.plugins.openmm.physicsml_potential import (
    PhysicsMLPotentialImplFactory,  # noqa F401
)
from openmmtools.utils import get_fastest_platform

from guardowl.protocols import (
    BondProfileProtocol,
    DOFTestParameters,
    MultiTemperatureProtocol,
    PropagationProtocol,
    StabilityTestParameters,
)
from guardowl.simulation import SystemFactory
from guardowl.testsystems import (
    HipenTestsystemFactory,
    SmallMoleculeTestsystemFactory,
    WaterboxTestsystemFactory,
)
from guardowl.utils import get_available_nnps_and_implementation

NNP_MODELS = [
    {},
    {},
    {},
    {},
]


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_setup_vacuum_protocol_individual_parts(model_properties: Dict) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()
    name = "ZINC00107550"

    nnp = model_properties["model"]
    implementation = "SPICE"

    testsystem = HipenTestsystemFactory().generate_testsystems(name)
    nnp_instance = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    output_folder = "test_stability_protocol"
    log_file_name = f"vacuum_{name}_{nnp}_{implementation}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    stability_test = MultiTemperatureProtocol()

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )

    params = StabilityTestParameters(
        protocol_length=5,
        temperature=[300, 400],
        ensemble="NVT",
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
        env="vacuum",
    )

    stability_test.perform_stability_test(params)


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_run_vacuum_protocol(model_properties: Dict) -> None:
    from guardowl.protocols import run_hipen_protocol

    nnp = model_properties["model"]
    implementation = "SPICE"

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )
    platform = get_fastest_platform()
    output_folder = "test_stability_protocol"

    run_hipen_protocol(
        1,
        nnp,
        implementation,
        300,
        reporter,
        platform,
        output_folder,
        nr_of_simulation_steps=2,
        model_properties=model_properties,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_setup_waterbox_protocol_individual_parts(
    ensemble: str, model_properties: Dict
) -> None:
    """Test if we can run a simulation for a number of steps"""

    nnp = model_properties["model"]
    implementation = "SPICE"

    # ---------------------------#
    platform = get_fastest_platform()

    edge_size = 5
    testsystem = WaterboxTestsystemFactory().generate_testsystems(
        edge_size * unit.angstrom, nr_of_equilibrium_steps=10
    )

    nnp_instance = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    output_folder = "test_stability_protocol"
    log_file_name = f"waterbox_{edge_size}A_{nnp}_{implementation}_{ensemble}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    stability_test = PropagationProtocol()

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )

    params = StabilityTestParameters(
        protocol_length=5,
        temperature=300,
        ensemble=ensemble,
        simulated_annealing=False,
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        state_data_reporter=reporter,
        env="solution",
    )

    stability_test.perform_stability_test(
        params,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_run_waterbox_protocol(ensemble: str, model_properties: Dict) -> None:
    from guardowl.protocols import run_waterbox_protocol

    nnp = model_properties["model"]
    implementation = "SPICE"

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )
    platform = get_fastest_platform()
    output_folder = "test_stability_protocol"

    run_waterbox_protocol(
        10,
        ensemble,
        nnp,
        implementation,
        300,
        reporter,
        platform,
        output_folder,
        nr_of_simulation_steps=2,
        nr_of_equilibrium_steps=10,
        model_properties=model_properties,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_run_alanine_dipeptide_protocol(ensemble: str, model_properties: Dict) -> None:
    from guardowl.protocols import run_alanine_dipeptide_protocol

    nnp = model_properties["model"]
    implementation = "SPICE"

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )
    platform = get_fastest_platform()
    output_folder = "test_stability_protocol"

    run_alanine_dipeptide_protocol(
        nnp,
        implementation,
        300,
        reporter,
        platform,
        output_folder,
        ensemble=ensemble,
        nr_of_simulation_steps=2,
        env="vacuum",
        model_properties=model_properties,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_run_pure_liquid_protocol(ensemble: str, model_properties: Dict) -> None:
    from guardowl.protocols import run_pure_liquid_protocol

    nnp = model_properties["model"]
    implementation = "SPICE"

    reporter = StateDataReporter(
        file=None,  # it is necessary to set this to None since it otherwise can't be passed to mp
        reportInterval=1,
        step=True,  # must be set to true
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        volume=True,
        speed=True,
    )
    platform = get_fastest_platform()
    output_folder = "test_stability_protocol"

    run_pure_liquid_protocol(
        nnp=nnp,
        implementation=implementation,
        temperature=300,
        reporter=reporter,
        platform=platform,
        output_folder=output_folder,
        molecule_name="ethane",
        nr_of_molecule=10,
        ensemble=ensemble,
        nr_of_simulation_steps=2,
        nr_of_equilibration_steps=10,
        model_properties=model_properties,
    )


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_DOF_protocol(model_properties: Dict) -> None:
    """Test if we can run a simulation for a number of steps"""

    nnp = model_properties["model"]
    implementation = "SPICE"

    # ---------------------------#
    platform = get_fastest_platform()

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems(name="ethanol")

    nnp_instance = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    output_folder = "test_stability_protocol"
    log_file_name = f"vacuum_{testsystem.testsystem_name}_{nnp}_{implementation}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    stability_test = BondProfileProtocol()
    params = DOFTestParameters(
        system=system,
        platform=platform,
        testsystem=testsystem,
        output_folder=output_folder,
        log_file_name=log_file_name,
        bond=[0, 3],
    )

    stability_test.perform_bond_scan(params)
