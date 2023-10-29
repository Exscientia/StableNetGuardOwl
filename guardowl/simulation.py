import logging
from typing import List, Type

from loguru import logger as log
from openmm import LangevinIntegrator, Platform, System, unit
from openmm.app import Simulation, Topology
from openmmml import MLPotential
from openmmtools.integrators import BAOABIntegrator

from .constants import collision_rate, stepsize


class SimulationFactory:
    @staticmethod
    def create_simulation(
        system: System,
        topology: Topology,
        platform: Platform,
        temperature: unit.Quantity,
    ) -> Simulation:
        """
        Create and return an OpenMM simulation instance using LangevinIntegrator.

        Parameters
        ----------
        system : System
            The OpenMM system object.
        topology : Topology
            The OpenMM topology object.
        platform : Platform
            The OpenMM Platform object for simulation.
        temperature : unit.Quantity
            The temperature at which to run the simulation.

        Returns
        -------
        Simulation
            The OpenMM simulation instance.

        """
        integrator = LangevinIntegrator(temperature, collision_rate, stepsize)

        return Simulation(
            topology,
            system,
            integrator,
            platform=platform,
        )

    @staticmethod
    def create_nvt_simulation(
        system: System,
        topology: Topology,
        platform: Platform,
        temperature: unit.Quantity,
    ) -> Simulation:
        """
        Create and return an OpenMM NVT simulation instance using BAOABIntegrator.

        Parameters
        ----------
        system : System
            The OpenMM system object.
        topology : Topology
            The OpenMM topology object.
        platform : Platform
            The OpenMM Platform object for simulation.
        temperature : unit.Quantity
            The temperature at which to run the simulation.

        Returns
        -------
        Simulation
            The OpenMM simulation instance.

        """

        integrator = BAOABIntegrator(temperature, collision_rate, stepsize)

        return Simulation(
            topology,
            system,
            integrator,
            platform=platform,
        )


class SystemFactory:
    @staticmethod
    def initialize_pure_ml_system(
        potential: Type[MLPotential],
        topology: Topology,
        remove_constraints: bool = True,
        implementation: str = "",
    ) -> System:
        """
        Initialize an OpenMM system using a machine learning potential.

        Parameters
        ----------
        potential : Type[MLPotential]
            The machine learning potential class.
        topology : Topology
            The OpenMM topology object.
        remove_constraints : bool, optional
            Whether to remove constraints from the system, by default True.
        implementation : str, optional
            The specific implementation of the ML potential, by default "".

        Returns
        -------
        System
            The OpenMM System object.

        """
        # create system & simulation instance
        if implementation:
            return potential.createSystem(
                topology,
                implementation=implementation,
                removeConstraints=remove_constraints,
                constraints=None,
                rigidWater=False,
            )

        return potential.createSystem(
            topology,
            removeConstraints=remove_constraints,
            constraints=None,
            rigidWater=False,
        )

    @staticmethod
    def initialize_mixed_ml_system(
        system: System,
        potential: Type[MLPotential],
        topology: Topology,
        interpolate: bool,
        ml_atoms: List[int],
        remove_constraints: bool = True,
        implementation: str = "",
    ) -> System:
        """
        Initialize an OpenMM system using both QML and MM potentials.

        Parameters
        ----------
        system : System
            The existing OpenMM System object.
        potential : Type[MLPotential]
            The machine learning potential class.
        topology : Topology
            The OpenMM topology object.
        interpolate : bool
            Whether to interpolate between the QML and MM potentials.
        ml_atoms : List[int]
            List of atom indices for which the ML potential will be applied.
        remove_constraints : bool, optional
            Whether to remove constraints from the system, by default True.
        implementation : str, optional
            The specific implementation of the ML potential, by default "".

        Returns
        -------
        System
            The OpenMM System object.

        """
        # create system & simulation instance
        if implementation:
            return potential.createMixedSystem(
                topology=topology,
                system=system,
                implementation=implementation,
                atoms=ml_atoms,
                removeConstraints=remove_constraints,
                interpolate=interpolate,
                constraints=None,
                rigidWater=False,
            )

        return potential.createMixedSystem(
            topology=topology,
            system=system,
            atoms=ml_atoms,
            removeConstraints=remove_constraints,
            interpolate=interpolate,
            constraints=None,
            rigidWater=False,
        )
