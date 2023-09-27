import numpy as np
import pytest
from openff.units.openmm import to_openmm
from openmm import unit
from openmm.app import DCDReporter, PDBFile
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform

from stability_test.setup import create_system_from_mol, generate_molecule
from stability_test.simulation import SimulationFactory, SystemFactory
from stability_test.testsystems import hipen_systems


@pytest.mark.parametrize(
    "nnp, e_lamb_0, e_lamb_1", [("ani2x", 295.1235918998718, -2346060.437261855)]
)
def test_generate_simulation_instance(
    nnp: str, e_lamb_0: float, e_lamb_1: float
) -> None:
    """Test if we can generate a simulation instance"""

    # set up system and topology and define ml region
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    mol = generate_molecule(smiles)
    system, topology = create_system_from_mol(mol)
    topology = topology.to_openmm()
    platform = get_fastest_platform()
    qml = MLPotential(nnp)
    ########################################################
    ########################################################
    # create MM simulation
    rsim = SimulationFactory.create_simulation(
        system, topology, platform=platform, temperature=unit.Quantity(300, unit.kelvin)
    )
    rsim.context.setPositions(to_openmm(mol.conformers[0]))
    e_sim_mm_endstate = (
        rsim.context.getState(getEnergy=True)
        .getPotentialEnergy()  # pylint: disable=unexpected-keyword-arg
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(e_sim_mm_endstate)
    assert np.isclose(e_sim_mm_endstate, e_lamb_0)
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
        temperature=unit.Quantity(300, unit.kelvin),
    )
    rsim.context.setPositions(to_openmm(mol.conformers[0]))
    e_sim_mm_endstate = (
        rsim.context.getState(getEnergy=True)
        .getPotentialEnergy()  # pylint: disable=unexpected-keyword-arg
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(e_sim_mm_endstate)
    assert np.isclose(e_sim_mm_endstate, e_lamb_1)
    # test minimization
    rsim.minimizeEnergy(maxIterations=1000)
    pos = rsim.context.getState(
        getPositions=True
    ).getPositions()  # pylint: disable=unexpected-keyword-arg
    with open("initial_frame_lamb_1.0.pdb", "w") as f:
        PDBFile.writeFile(topology, pos, f)

@pytest.mark.parametrize(
    "nnp, implementation", [("ani2x", "nnpops"), ("ani2x", "torchani"), ("ani2x", "")]
)
def test_simulating(nnp: str, implementation: str) -> None:
    """Test if we can run a simulation for a number of steps"""

    # set up system and topology and define ml region
    name = list(hipen_systems.keys())[1]
    smiles = hipen_systems[name]
    mol = generate_molecule(smiles)
    system, topology = create_system_from_mol(mol)
    qml = MLPotential(nnp)
    platform = get_fastest_platform()
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
        platform=platform,
        temperature=unit.Quantity(300, unit.kelvin),
    )
    # set position
    qsim.context.setPositions(to_openmm(mol.conformers[0]))
    # simulate
    qsim.reporters.append(DCDReporter("test.dcd", 10))
    qsim.step(100)
    del qsim
