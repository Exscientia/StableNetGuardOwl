import logging
import time
import timeit
import warnings
from multiprocessing import Event, Process, Value
from typing import Generator

import nvidia_smi
import torch
from openmm import unit
from openmm.app import Simulation
from openmmml import MLPotential
from openmmtools.testsystems import TestSystem

TIMEOUT_SECONDS = 200
from .simulation import SimulationFactory, SystemFactory

warnings.filterwarnings("ignore")
log = logging.getLogger("benchmark")


class GPUMemoryLogger(Process):
    """
    Process that logs the GPU memory usage of the GPU that is used by the simulation.
    """

    def __init__(self, caller, max_gpu_memory) -> None:  # type: ignore
        Process.__init__(self)
        self.caller = caller
        self.max_gpu_memory = max_gpu_memory

    def run(self) -> None:
        # get gpu momory footprint
        list_of_memory = []
        device_id = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        while not self.caller.stop_flag.is_set():
            time.sleep(0.1)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            mem = info.used
            list_of_memory.append(mem)
            if self.caller.report_flag.is_set():
                self.max_gpu_memory.value = max(list_of_memory)
                self.caller.report_flag.clear()
                log.info("reported ...")
                log.info(f"Max GPU memory: {self.max_gpu_memory.value}")
                # reset list
                list_of_memory = []

        log.info("exiting")


class PerformTest(Process):
    """
    Process that performs a single point energy calculation for a given NNP and a given test system.
    That also includes setting up the simulation and the system.
    """

    def __init__(  # type: ignore
        self,
        nnp: str,
        testsystem: TestSystem,
        remove_constraints: bool,
        platform: str,
        qml_timing,
        reference_timing,
    ) -> None:
        Process.__init__(self)
        self.simulation_factory = SimulationFactory()
        self.system_factory = SystemFactory()
        self.testsystem = testsystem
        self.nnp = nnp
        self.remove_constraints = remove_constraints
        self.platform = platform
        self.qml_timing = qml_timing
        self.reference_timing = reference_timing

    @staticmethod
    def get_timing_for_spe_calculation(
        sim: Simulation, testsystem: TestSystem
    ) -> float:
        sim.context.setPositions(testsystem.positions)
        # we do this twice since for the first calculation there is always an overhead
        sim.context.getState(getEnergy=True)
        return (
            timeit.timeit(
                stmt="sim.context.getState(getEnergy=True)",
                globals=locals(),
                number=10,
            )
            / 10
        )

    def run(self) -> None:
        # this is executed as soon as the process is started

        print(f"{self.platform=}")
        potential = MLPotential(self.nnp)

        system = self.system_factory.initialize_system(
            potential,
            self.testsystem.topology,
            self.remove_constraints,
        )

        psim = self.simulation_factory.create_simulation(
            system,
            self.testsystem.topology,
            platform=self.platform,
            temperature=unit.Quantity(300.0, unit.kelvin),
        )

        # -----------------------------------#
        # single point energy calculation
        self.qml_timing.value = self.get_timing_for_spe_calculation(
            psim, self.testsystem
        )

        # -----------------------------------#
        # get reference single point energy
        rsim = self.simulation_factory.create_simulation(
            self.testsystem.system,
            self.testsystem.topology,
            platform=self.platform,
            temperature=unit.Quantity(300.0, unit.kelvin),
        )
        self.reference_timing.value = self.get_timing_for_spe_calculation(
            rsim, self.testsystem
        )

        del psim
        del system
        del potential
        torch.cuda.empty_cache()


class Benchmark:
    """
    A class to benchmark the performance of a neural network potential.
    It creates two processes, one that tracks the GPU memory usage and one that performs the benchmark.
    """

    def __init__(
        self,
    ) -> None:
        self.reference_timing, self.qml_timing, self.gpu_memory = [], [], []  # type: ignore
        # initialize flags --- stop flag is used to end the GPU memory monitoring, report flag is used to report the current gpu memory
        self.stop_flag = Event()
        self.report_flag = Event()
        # initialize shared memory for max gpu memory
        self.max_gpu_mem = Value("d", 0.0, lock=False)
        self.qml_timing = []
        self.reference_timing = []
        # initialize GPUMemoryLogger
        self.gpu_logger = GPUMemoryLogger(self, self.max_gpu_mem)

    def run_benchmark(
        self,
        nnp: str,
        testsystems: Generator[TestSystem, None, None],
        remove_constraints: bool,
        platform: str,
    ) -> None:
        self.reference_timing, self.qml_timing, self.gpu_memory = [], [], []
        # start memory logger
        self.gpu_logger.start()
        # clear flags
        self.stop_flag.clear()

        for testsystem in testsystems:
            self.report_flag.clear()
            # initialize shared memory for qml_timing and reference_timing
            qml_timing, reference_timing = Value("d", 0.0, lock=False), Value(
                "d", 0.0, lock=False
            )
            # start benchmark
            simulation_test = PerformTest(
                nnp,
                testsystem,
                remove_constraints,
                platform,
                qml_timing,
                reference_timing,
            )
            simulation_test.start()
            log.info("Started simulation")
            # wait for simulation to finish and retrieve results
            simulation_test.join(TIMEOUT_SECONDS)
            simulation_test.terminate()
            # set flags to retrieve results
            log.info("Finished simulation")
            self.report_flag.set()
            # retrieve results
            log.info("Retrieved results")
            time.sleep(1)
            print(self.max_gpu_mem.value)
            if int(self.max_gpu_mem.value) == 0:
                print("GPU process ran out of memory")
            else:
                self.gpu_memory.append(self.max_gpu_mem.value)
                log.debug(self.gpu_memory)
            self.qml_timing.append(qml_timing.value)
            self.reference_timing.append(reference_timing.value)
            log.debug(self.qml_timing)
            log.debug(self.reference_timing)

        # stop memory logger
        self.stop_flag.set()
        self.gpu_logger.join(10)
        self.gpu_logger.terminate()
