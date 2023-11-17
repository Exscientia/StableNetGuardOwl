from typing import Dict
import numpy as np
import pytest
from openff.units.openmm import to_openmm
from openmm import unit
from openmm.app import DCDReporter, PDBFile
from exs.physicsml.plugins.openmm.mlpotential import MLPotential
from exs.physicsml.plugins.openmm.physicsml_potential import (
    PhysicsMLPotentialImplFactory,  # noqa F401
)
from openmmtools.utils import get_fastest_platform

from guardowl.simulation import SimulationFactory, SystemFactory
from guardowl.utils import (
    get_available_nnps_and_implementation,
    gpu_memory_constrained_nnps_and_implementation,
)


NNP_MODELS = [
    {},
    {},
    {},
    {},
]


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_generate_simulation_instance(
    model_properties: Dict, generate_hipen_system
) -> None:
    """Test if we can generate a simulation instance"""

    # set up system and topology and define ml region
    system, topology, mol = generate_hipen_system
    topology = topology.to_openmm()
    platform = get_fastest_platform()

    qml = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )

    ########################################################
    ########################################################
    # create MM simulation
    rsim = SimulationFactory.create_simulation(
        system,
        topology,
        platform=platform,
        env="vacuum",
        ensemble="NVT",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    rsim.context.setPositions(to_openmm(mol.conformers[0]))
    e_sim_mm_endstate = (
        rsim.context.getState(getEnergy=True)
        .getPotentialEnergy()  # pylint: disable=unexpected-keyword-arg
        .value_in_unit(unit.kilojoule_per_mole)
    )

    assert np.isclose(e_sim_mm_endstate, model_properties["mm_value"])

    ########################################################
    ########################################################
    # create ML simulation
    rsim = SimulationFactory.create_simulation(
        SystemFactory().initialize_pure_ml_system(
            qml,
            topology,
        ),
        topology,
        platform=platform,
        ensemble="NVT",
        env="vacuum",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    rsim.context.setPositions(to_openmm(mol.conformers[0]))
    e_sim_mm_endstate = (
        rsim.context.getState(getEnergy=True)
        .getPotentialEnergy()  # pylint: disable=unexpected-keyword-arg
        .value_in_unit(unit.kilojoule_per_mole)
    )

    assert np.isclose(e_sim_mm_endstate, model_properties["ml_value"])

    # test minimization
    rsim.minimizeEnergy(maxIterations=1000)

    pos = rsim.context.getState(
        getPositions=True
    ).getPositions()  # pylint: disable=unexpected-keyword-arg
    with open("initial_frame_lamb_1.0.pdb", "w") as f:
        PDBFile.writeFile(topology, pos, f)


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_simulating(model_properties: Dict, generate_hipen_system) -> None:
    """Test if we can run a simulation for a number of steps"""

    # set up system and topology and define ml region
    system, topology, mol = generate_hipen_system
    platform = get_fastest_platform()

    implementation = "SPICE"

    qml = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )

    topology = topology.to_openmm()
    ########################################################
    # ---------------------------#
    # generate pure ML simulation
    qsim = SimulationFactory.create_simulation(
        SystemFactory().initialize_pure_ml_system(
            qml,
            topology,
            implementation=implementation,
        ),
        topology,
        env="vacuum",
        platform=platform,
        ensemble="NVT",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    # set position
    qsim.context.setPositions(to_openmm(mol.conformers[0]))
    # simulate
    qsim.reporters.append(DCDReporter("test.dcd", 10))
    qsim.step(5)
    del qsim


@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_pure_liquid_simulation(model_properties: Dict):
    from guardowl.testsystems import PureLiquidTestsystemFactory

    factory = PureLiquidTestsystemFactory()
    liquid_box = factory.generate_testsystems(
        name="ethane", nr_of_copies=150, nr_of_equilibration_steps=500
    )

    platform = get_fastest_platform()

    implementation = "SPICE"

    qml = MLPotential(
        "physicsml_model",
        repo_url="git@bitbucket.org:exscientia/qmml-experiments.git",
        rev=model_properties["rev"],
        model_path_in_repo=model_properties["model_path_in_repo"],
        precision=model_properties["precision"],
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform.getName().lower(),
    )
    platform = get_fastest_platform()
    ########################################################
    # ---------------------------#
    # generate pure ML simulation
    qsim = SimulationFactory.create_simulation(
        SystemFactory().initialize_pure_ml_system(
            qml,
            liquid_box.topology,
            implementation=implementation,
        ),
        liquid_box.topology,
        env="solution",
        platform=platform,
        ensemble="NpT",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    # set position
    qsim.context.setPositions(liquid_box.positions)
    # simulate
    qsim.reporters.append(DCDReporter("test.dcd", 10))
    qsim.step(5)
