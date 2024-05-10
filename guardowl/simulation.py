import logging
from typing import List, Literal, Optional, Type

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
        env: Literal["vacuum", "solution"],
        device_index: int = 0,
        ensemble: str = "NVT",
    ) -> Simulation:
        """
        Creates an OpenMM Simulation object with specified parameters.

        Parameters
        ----------
        system : System
            The OpenMM system object.
        topology : Topology
            The OpenMM topology object.
        platform : Platform
            The OpenMM Platform object for simulation.
        temperature : unit.Quantity
            The temperature for the simulation.
        env : str
            The environment of the simulation ("vacuum" or "solution").
        device_index : int, optional
            GPU device index for CUDA platform, by default 0.
        ensemble : str, optional
            The ensemble for the simulation ("NVT" or "NPT"), by default "NVT".

        Returns
        -------
        Simulation
            Configured OpenMM Simulation object.
        """
        from openmm import MonteCarloBarostat

        if ensemble.lower() == "nve":
            integrator = BAOABIntegrator(temperature, collision_rate, stepsize)
        else:
            integrator = LangevinIntegrator(temperature, collision_rate, stepsize)

        if ensemble.casefold() == "npt" and env != "vacuum":  # for NpT add barostat
            barostate_force_id = system.addForce(
                MonteCarloBarostat(unit.Quantity(1, unit.atmosphere), temperature)
            )

        if platform.getName() == "CUDA":
            prop = {"CudaDeviceIndex", str(device_index), "CudaPrecision", "mixed"}
            simulation = Simulation(topology, system, integrator, platform, prop)
        else:
            simulation = Simulation(topology, system, integrator, platform)

        return simulation


class SystemFactory:
    @staticmethod
    def initialize_system(
        potential: Type[MLPotential],
        topology: Topology,
    ) -> System:
        """
        Initialize an OpenMM system using a machine learning potential.

        Parameters
        ----------
        potential : Type[MLPotential]
            The machine learning potential class.
        topology : Topology
            The OpenMM topology object.

        Returns
        -------
        System
            The OpenMM System object.

        Examples
        --------
        >>> potential = MLPotential
        >>> topology = Topology()
        >>> system = SystemFactory.initialize_system(potential, topology)
        """
        return potential.createSystem(topology, implementation="torchani")
