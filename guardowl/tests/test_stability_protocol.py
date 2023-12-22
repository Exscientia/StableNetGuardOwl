from pathlib import Path

import pytest
from openmm import unit
from openmm.app import StateDataReporter
from openmmml import MLPotential
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
    SmallMoleculeTestsystemFactory,
    WaterboxTestsystemFactory,
)
from guardowl.utils import get_available_nnps_and_implementation


@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_setup_vacuum_protocol_individual_parts(nnp: str, implementation: str) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()
    name = "ZINC00107550"

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems_from_name(name)
    nnp_instance = MLPotential(nnp)

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


@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_run_vacuum_protocol(nnp: str, implementation: str) -> None:
    from guardowl.protocols import run_hipen_protocol

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
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_setup_waterbox_protocol_individual_parts(
    ensemble: str, nnp: str, implementation: str, temperature: int = 300
) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()

    edge_size = 5
    testsystem = WaterboxTestsystemFactory().generate_testsystems(
        edge_size * unit.angstrom, nr_of_equilibrium_steps=10
    )
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_pure_ml_system(
        nnp_instance,
        testsystem.topology,
        implementation=implementation,
    )

    output_folder = "test_stability_protocol"
    log_file_name = (
        f"waterbox_{edge_size}A_{nnp}_{implementation}_{ensemble}_{temperature}K"
    )
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
        temperature=temperature,
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
@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_run_waterbox_protocol(ensemble: str, nnp: str, implementation: str) -> None:
    from guardowl.protocols import run_waterbox_protocol

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
        5,
        ensemble,
        nnp,
        implementation,
        300,
        reporter,
        platform,
        output_folder,
        nr_of_simulation_steps=2,
        nr_of_equilibrium_steps=10,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_run_alanine_dipeptide_protocol(
    ensemble: str, nnp: str, implementation: str
) -> None:
    from guardowl.protocols import run_alanine_dipeptide_protocol

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
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_run_pure_liquid_protocol(ensemble: str, nnp: str, implementation: str) -> None:
    from guardowl.protocols import run_pure_liquid_protocol

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
    )


@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_DOF_protocol(nnp: str, implementation: str) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()

    testsystem = SmallMoleculeTestsystemFactory().generate_testsystems(name="ethanol")

    nnp_instance = MLPotential(nnp)

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


def test_input_generation_for_minimization_tests():
    from guardowl.utils import (
        _generate_input_for_minimization_test,
        _generate_file_list_for_minimization_test,
        extract_drugbank_tar_gz,
    )
    import numpy as np

    # extract tar.gz data
    extract_drugbank_tar_gz()
    # read in file names and build Iterators
    files = _generate_file_list_for_minimization_test()
    # read in first file
    (minimized_file, minimized_position), (start_file, start_position) = next(
        _generate_input_for_minimization_test(files)
    )
    # test if the file base is the same
    assert (
        minimized_file
        == "/home/mwieder/Work/Projects/StableNetGuardOwl/guardowl/data/drugbank/owl/11117974/orca_input.xyz"
    )
    assert "".join(minimized_file.split("/")[:-1]) == "".join(
        start_file.split("/")[:-1]
    )

    assert np.allclose(
        minimized_position[0], [-1.33526786531436, 2.48368695037195, 1.04890049484746]
    )

    # now shuffel
    files = _generate_file_list_for_minimization_test(shuffel=True)
    (minimized_file, minimized_position), (start_file, start_position) = next(
        _generate_input_for_minimization_test(files)
    )
    assert not (
        minimized_file
        == "/home/mwieder/Work/Projects/StableNetGuardOwl/guardowl/data/drugbank/owl/11117974/orca_input.xyz"
    )
    assert "".join(minimized_file.split("/")[:-1]) == "".join(
        start_file.split("/")[:-1]
    )


@pytest.mark.parametrize("nnp, implementation", get_available_nnps_and_implementation())
def test_run_detect_minimum_test(nnp, implementation, extracted_dir):
    from guardowl.protocols import run_detect_minimum_test

    platform = get_fastest_platform()

    run_detect_minimum_test(nnp, implementation, platform, extracted_dir, percentage=10)
