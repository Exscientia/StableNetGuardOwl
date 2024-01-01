from typing import TextIO, Tuple
from openmm import State, System, unit, Platform
from openmm.app import Simulation, StateDataReporter, Topology

# StateDataReporter with custom print function
class ContinuousProgressReporter(object):
    """A class for reporting the progress of a simulation continuously.

    Parameters
    ----------
    iostream : TextIO
        Output stream to write the progress report to.
    total_steps : int
        Total number of steps in the simulation.
    reportInterval : int
        Interval at which to report the progress.

    Attributes
    ----------
    _out : TextIO
        The output stream.
    _reportInterval : int
        The report interval.
    _total_steps : int
        Total number of steps.
    """

    def __init__(self, iostream: TextIO, total_steps: int, reportInterval: int):
        self._out = iostream
        self._reportInterval = reportInterval
        self._total_steps = total_steps

    def describeNextReport(self, simulation: Simulation) -> Tuple:
        """
        Returns information about the next report.

        Parameters
        ----------
        simulation : Simulation
            The simulation to report on.

        Returns
        -------
        Tuple
            A tuple containing information about the next report.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation: Simulation, state: State) -> None:
        """
        Reports the progress of the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation to report on.
        state : State
            The state of the simulation.
        """
        progress = 100.0 * simulation.currentStep / self._total_steps
        self._out.write(f"\rProgress: {progress:.2f}")
        self._out.flush()
