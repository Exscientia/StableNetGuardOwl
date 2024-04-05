from pathlib import Path

import pytest
from guardowl.protocols import (
    BondProfileProtocol,
    DOFTestParameters,
    MultiTemperatureProtocol,
    PropagationProtocol,
    StabilityTestParameters,
)
from guardowl.simulation import SystemFactory
from guardowl.testsystems import (
    LiquidOption,
    SmallMoleculeVacuumOption,
    TestsystemFactory,
)
from guardowl.utils import get_available_nnps
from openmm import unit
from openmm.app import StateDataReporter
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform
from guardowl.setup import PotentialFactory


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_setup_vacuum_protocol_individual_parts(nnp: str) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()
    name = "ZINC00107550"
    opt = SmallMoleculeVacuumOption(
        name=name,
    )

    testsystem = TestsystemFactory().generate_testsystem(opt)
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_system(
        nnp_instance,
        testsystem.topology,
    )
    output_folder = "test_stability_protocol"
    log_file_name = f"vacuum_{name}_{nnp}"
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


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_run_vacuum_protocol(nnp: str) -> None:
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
        300,
        reporter,
        platform,
        output_folder,
        nr_of_simulation_steps=2,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("nnp", get_available_nnps())
def test_setup_waterbox_protocol_individual_parts(
    ensemble: str, nnp: str, temperature: int = 300
) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()
    edge_length = 5
    opt = LiquidOption(name="water", edge_length=edge_length * unit.angstrom)
    testsystem = TestsystemFactory().generate_testsystem(opt)
    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_system(
        nnp_instance,
        testsystem.topology,
    )

    output_folder = "test_stability_protocol"
    log_file_name = f"waterbox_{edge_length}A_{nnp}_{ensemble}_{temperature}K"
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


from typing import Dict, Tuple


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("params", get_available_nnps())
def test_run_waterbox_protocol(
    ensemble: str, params: Dict[str, Tuple[str, int, float]]
) -> None:
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
    nnp = PotentialFactory().initialize_potential(params)

    run_waterbox_protocol(
        5,
        ensemble,
        nnp,
        300,
        reporter,
        platform,
        output_folder,
        nr_of_simulation_steps=2,
        nr_of_equilibrium_steps=10,
    )


@pytest.mark.parametrize("environment", ["vacuum", "solution"])
@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("params", get_available_nnps())
def test_run_alanine_dipeptide_protocol(
    environment: str, ensemble: str, params: Dict[str, Tuple[str, int, float]]
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
    nnp = PotentialFactory().initialize_potential(params)
    run_alanine_dipeptide_protocol(
        nnp,
        300,
        reporter,
        platform,
        output_folder,
        ensemble=ensemble,
        nr_of_simulation_steps=2,
        env=environment,
    )


@pytest.mark.parametrize("ensemble", ["NVE", "NVT", "NpT"])
@pytest.mark.parametrize("params", get_available_nnps())
def test_run_pure_liquid_protocol(
    ensemble: str, params: Dict[str, Tuple[str, int, float]]
) -> None:
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
    nnp = PotentialFactory().initialize_potential(params)

    run_pure_liquid_protocol(
        nnp=nnp,
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


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_DOF_protocol(nnp: str) -> None:
    """Test if we can run a simulation for a number of steps"""

    # ---------------------------#
    platform = get_fastest_platform()

    opt = SmallMoleculeVacuumOption(name="ethanol")
    testsystem = TestsystemFactory().generate_testsystem(opt)

    nnp_instance = MLPotential(nnp)

    system = SystemFactory().initialize_system(
        nnp_instance,
        testsystem.topology,
    )

    output_folder = "test_stability_protocol"
    log_file_name = f"vacuum_{opt.name}_{nnp}"
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

    stability_test.perform_scan(params)


import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Github Actions does not return the same file order"
)
def test_input_generation_for_minimization_tests():
    import numpy as np
    from guardowl.utils import (
        _generate_file_list_for_minimization_test,
        _generate_input_for_minimization_test,
        extract_drugbank_tar_gz,
    )

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
        "/".join(minimized_file.split("/")[-6:])
        == "guardowl/data/drugbank/owl/49957/orca_input.xyz"
    )
    assert "".join(minimized_file.split("/")[:-1]) == "".join(
        start_file.split("/")[:-1]
    )

    assert np.allclose(minimized_position[0], [5.07249404, -0.21016912, -0.0933702])

    # now shuffel
    files = _generate_file_list_for_minimization_test(shuffle=True)
    (minimized_file, minimized_position), (start_file, start_position) = next(
        _generate_input_for_minimization_test(files)
    )

    assert not (
        "/".join(minimized_file.split("/")[-6:])
        == "guardowl/data/drugbank/owl/49957/orca_input.xyz"
    )
    assert "".join(minimized_file.split("/")[:-1]) == "".join(
        start_file.split("/")[:-1]
    )
    # generate mol from sdf file
    sdf_file = "".join(start_file.split(".")[0]) + ".sdf"
    opt = SmallMoleculeVacuumOption(path=sdf_file)

    reference_testsystem = TestsystemFactory().generate_testsystem(opt)
    # set positions
    reference_testsystem.positions = minimized_position


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_run_detect_minimum(nnp, tmp_dir):
    from guardowl.protocols import run_detect_minimum

    platform = get_fastest_platform()
    nnp_instance = MLPotential(nnp)

    run_detect_minimum(
        nnp_instance,
        platform,
        tmp_dir,
        percentage=0.1,
        skip_molecules_above_heavy_atom_threshold=8,
    )
