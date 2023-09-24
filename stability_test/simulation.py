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
        Returns an encapsulated OpenMM simulation instance


        Args:
            system (System)
            topology (Topology)
            platform (str)

        Returns:
            _type_: PotentialSimulation
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
        Returns an encapsulated OpenMM simulation instance


        Args:
            system (System)
            topology (Topology)
            platform (str)

        Returns:
            _type_: PotentialSimulation
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
        """Returns an OpenMM simulation instance with a QML potential energy function

        Args:
            potential (Type[MLPotential])
            topology (Topology)
            remove_constraints (bool, optional): Defaults to True.
            platform (str, optional): Defaults to "CUDA".
            implementation (str, optional): Defaults to "".

        Returns:
            Simulation
        """
        # create system & simulation instance
        if implementation:
            return potential.createSystem(
                topology,
                implementation=implementation,
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
        Returns an OpenMM simulation instance with a QML and MM potential energy function


        Args:
            system (System)
            potential (Type[MLPotential])
            topology (Topology)
            interpolate (bool)
            ml_atoms (List[int])
            remove_constraints (bool, optional): Defaults to True.
            platform (str, optional): Defaults to "CUDA".
            implementation (str, optional): Defaults to "".

        Returns:
            Simulation
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

    def initialize_interaction_interpolation_ml_system(self) -> None:
        pass
