from typing import Literal, Tuple

import numpy as np
import pytest
from guardowl.setup import PotentialFactory
from guardowl.simulation import SimulationFactory, SystemFactory
from guardowl.utils import get_available_nnps
from openmm import unit
from openmm.app import DCDReporter, PDBFile
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform

from typing import Dict, Tuple


@pytest.mark.parametrize("params", get_available_nnps())
def test_generate_simulation_instance(
    params: Dict[str, Tuple[str, int, float]],
    single_hipen_system: PDBFile,
) -> None:
    """Test if we can generate a simulation instance"""

    # set up system and topology and define ml region
    pdb = single_hipen_system
    platform = get_fastest_platform()
    nnp = PotentialFactory().initialize_potential(params)
    ########################################################
    ########################################################
    # create ML simulation
    sim = SimulationFactory.create_simulation(
        SystemFactory.initialize_system(
            nnp,
            pdb.topology,
        ),
        pdb.topology,
        platform=platform,
        ensemble="NVT",
        env="vacuum",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    sim.context.setPositions(pdb.positions)
    e = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    # test minimization
    sim.minimizeEnergy(maxIterations=100)
    pos = sim.context.getState(getPositions=True).getPositions()


from typing import Any, Dict


@pytest.mark.parametrize("params", get_available_nnps())
def test_simulating(
    params: Dict[str, Tuple[str, int, float]],
    single_hipen_system: PDBFile,
) -> None:
    """Test if we can run a simulation for a number of steps"""

    # set up system and topology and define ml region
    pdb = single_hipen_system
    nnp = PotentialFactory().initialize_potential(params)
    platform = get_fastest_platform()

    ########################################################
    # ---------------------------#
    # generate pure ML simulation
    sim = SimulationFactory.create_simulation(
        SystemFactory().initialize_system(
            nnp,
            pdb.topology,
        ),
        pdb.topology,
        env="vacuum",
        platform=platform,
        ensemble="NVT",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    # set position
    sim.context.setPositions(pdb.positions)
    # simulate
    sim.reporters.append(DCDReporter("test.dcd", 10))
    sim.step(5)
    del sim


@pytest.mark.parametrize("params", get_available_nnps())
def test_pure_liquid_simulation(
    params: Dict[str, Tuple[str, int, float]],
):
    from guardowl.testsystems import LiquidOption, TestsystemFactory

    opt = LiquidOption(name="ethane", nr_of_copies=150)

    factory = TestsystemFactory()
    liquid_box = factory.generate_testsystem(opt)
    nnp = PotentialFactory().initialize_potential(params)
    platform = get_fastest_platform()
    ########################################################
    # ---------------------------#
    # generate pure ML simulation
    sim = SimulationFactory.create_simulation(
        SystemFactory().initialize_system(
            nnp,
            liquid_box.topology,
        ),
        liquid_box.topology,
        env="solution",
        platform=platform,
        ensemble="NpT",
        temperature=unit.Quantity(300, unit.kelvin),
    )
    # set position
    sim.context.setPositions(liquid_box.positions)
    # simulate
    sim.reporters.append(DCDReporter("test.dcd", 10))
    sim.step(5)
