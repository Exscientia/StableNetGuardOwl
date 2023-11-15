from dataclasses import dataclass, field
from openmm import Platform, System, unit
from openmm.app import StateDataReporter
from openmmtools.testsystems import TestSystem
from typing import List, Union, Optional


@dataclass
class BaseParameters:
    system: System
    platform: Platform
    testsystem: TestSystem
    output_folder: str
    log_file_name: str


@dataclass
class MinimizationTestParameters(BaseParameters):
    convergence_criteria: unit.Quantity = field(
        default_factory=1.0 * unit.kilojoule_per_mole / unit.nanometer
    )


@dataclass
class StabilityTestParameters(BaseParameters):
    """Parameters for a stability test.

    Parameters are stored as attributes.

    Attributes
    ----------
    protocol_length : int
        Length of the protocol in time units.
    temperature : unit.Quantity
        Temperature of the simulation.
    ensemble : str
        Ensemble type ('NVT', 'NPT', etc.).
    simulated_annealing : bool
        Whether simulated annealing is to be used.
    system : System
        The OpenMM System object.
    platform : Platform
        The OpenMM Platform object.
    testsystem : TestSystem
        The test system for the simulation.
    output_folder : str
        Path to the output folder.
    log_file_name : str
        Name of the log file.
    state_data_reporter : StateDataReporter
        The OpenMM StateDataReporter object.
    """

    protocol_length: int
    temperature: unit.Quantity
    env: str
    simulated_annealing: bool
    state_data_reporter: StateDataReporter
    device_index: int = 0
    ensemble: Optional[str] = None


@dataclass
class DOFTestParameters(BaseParameters):
    """Parameters for a degree of freedom (DOF) test.

    In addition to attributes in StabilityTestParameters, extra attributes for DOF tests are included.

    Attributes
    ----------
    bond : List
        List of atom pairs to be considered as bonds.
    angle : List
        List of atom triplets to be considered as angles.
    torsion : List
        List of atom quartets to be considered as torsions.
    """

    bond: List = field(default_factory=lambda: [])
    angle: List = field(default_factory=lambda: [])
    torsion: List = field(default_factory=lambda: [])
