from dataclasses import dataclass, field
from openmm import Platform, System, unit
from openmm.app import StateDataReporter
from openmmtools.testsystems import TestSystem
from typing import List, Union, Optional, Literal, Literal
from .constants import Environment


@dataclass
class BaseParameters:
    """
    Base class for simulation parameters.

    Attributes
    ----------
    system : System
        The OpenMM System object to be used for the simulation.
    platform : Platform
        The OpenMM Platform object specifying the computation platform.
    testsystem : TestSystem
        The test system object containing the molecular system setup.
    output_folder : str
        Directory path where output files will be saved.
    log_file_name : str
        Filename for the log file.

    """

    system: System
    platform: Platform
    testsystem: TestSystem
    output_folder: str
    log_file_name: str


@dataclass
class MinimizationTestParameters(BaseParameters):
    """
    Parameters specific to minimization tests.

    Attributes
    ----------
    convergence_criteria : unit.Quantity
        The energy convergence criteria for the minimization process.
    env : Literal["vacuum", "solution"]
        The environment of the simulation (e.g., 'vacuum', 'solution').
    device_index : int
        Index of the GPU device to use for the simulation.
    temperature : unit.Quantity
        The temperature at which the simulation is performed.
    ensemble : str
        The statistical ensemble for the simulation (e.g., 'NVT').

    """

    convergence_criteria: unit.Quantity = field(
        default_factory=lambda: unit.Quantity(
            0.5, unit.kilojoule_per_mole / unit.angstrom
        )
    )
    env: Literal["vacuum"] = "vacuum"
    device_index: int = 0
    temperature: unit.Quantity = field(
        default_factory=lambda: unit.Quantity(300, unit.kelvin)
    )

    ensemble: str = "NVT"


from typing import Literal


@dataclass
class StabilityTestParameters(BaseParameters):
    """Parameters for a stability test.

    Parameters are stored as attributes.

    Attributes
    ----------
    protocol_length : int
        The duration of the test protocol.
    temperature : unit.Quantity
        The temperature at which the test is conducted.
    env : Literal["vacuum", "solution"]
        The environment for the simulation (e.g., 'vacuum', 'solution').
    simulated_annealing : bool
        Flag to indicate if simulated annealing is used.
    state_data_reporter : StateDataReporter
        The OpenMM StateDataReporter object for logging.
    device_index : int
        Index of the GPU device to use for the simulation.
    ensemble : Optional[str]
        The statistical ensemble for the simulation (e.g., 'NVT', 'NPT'). None if not applicable.

    """

    protocol_length: int
    temperature: unit.Quantity
    env: Literal["vacuum", "solution"]
    simulated_annealing: bool
    state_data_reporter: StateDataReporter
    device_index: int = 0
    ensemble: Literal["NVT", "NPT", "NVE", None] = None


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
