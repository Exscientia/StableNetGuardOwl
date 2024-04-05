# import pytest
# from openmm import unit
# from openmmtools.testsystems import WaterBox, WaterCluster

# from exs.quams.benchmark import Benchmark


# @pytest.mark.parametrize("nnp", [("ani2x")])
# def test_run_benchmark_for_watercluster(nnp: str)-> None:
#     ########################################
#     # Watercluster
#     ########################################
#     # create benchmark instance
#     benchmark = Benchmark()
#     # generate system to benchmark
#     watercluster = WaterCluster(n_waters=20)
#     _, topology = watercluster.system, watercluster.topology
#     ml_atoms = [atom.index for atom in topology.atoms()]
#     assert len(ml_atoms) / 3 == 20
#     # initialize ANI-2x
#     benchmark.run_benchmark(nnp, watercluster, False, platform="CUDA")


# @pytest.mark.parametrize(
#     "nnp, [("ani2x"), ("ani2x", "torchani")]
# )
# def test_run_benchmark_for_waterbox(nnp: str, implementation: str) -> None:
#     ########################################
#     # waterbox
#     ########################################
#     # create benchmark instance
#     benchmark = Benchmark()
#     # generate system to benchmark
#     watercluster_test_systems = (
#         WaterBox(
#             unit.Quantity(box_edge, unit.angstrom),
#             cutoff=unit.Quantity(5, unit.angstrom),
#         )
#         for box_edge in [10]  # , 15, 20]
#     )
#     benchmark.run_benchmark(
#         nnp,
#         watercluster_test_systems,
#         False,
#         platform="CUDA",
#         implementation=implementation,
#     )
