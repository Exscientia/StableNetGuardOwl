from typing import Literal, Tuple
import numpy as np
from openmm.openmm import System
import pytest
from openmm import unit
from openmm.app import DCDReporter, PDBFile
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform

from guardowl.simulation import SimulationFactory, SystemFactory
from guardowl.utils import (
    get_available_nnps,
)


@pytest.mark.parametrize("nnp, e_ref", [("ani2x", -2346020.730264931)])
def test_generate_simulation_instance(
    nnp: str,
    e_ref: float,
    single_hipen_system: PDBFile,
) -> None:
    """Test if we can generate a simulation instance"""

    # set up system and topology and define ml region
    pdb = single_hipen_system
    platform = get_fastest_platform()
    nnp = MLPotential(nnp)
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
    assert np.isclose(e, e_ref)
    # test minimization
    sim.minimizeEnergy(maxIterations=1000)
    pos = sim.context.getState(getPositions=True).getPositions()


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_simulating(
    nnp: str,
    single_hipen_system: PDBFile,
) -> None:
    """Test if we can run a simulation for a number of steps"""

    # set up system and topology and define ml region
    pdb = single_hipen_system
    qml = MLPotential(nnp)
    platform = get_fastest_platform()

    ########################################################
    # ---------------------------#
    # generate pure ML simulation
    sim = SimulationFactory.create_simulation(
        SystemFactory().initialize_system(
            qml,
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


@pytest.mark.parametrize("nnp", get_available_nnps())
def test_pure_liquid_simulation(
    nnp: str,
):
    from guardowl.testsystems import TestsystemFactory, LiquidOption

    opt = LiquidOption(name="ethane", nr_of_copies=150)

    factory = TestsystemFactory()
    liquid_box = factory.generate_testsystem(opt)
    nnp = MLPotential(nnp)
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
