from collections import namedtuple
from dataclasses import dataclass
import logging
import fire
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from openmmtools.testsystems import WaterCluster
from rich.progress import Progress

log = logging.getLogger("rich")


def plot_timing(benchmark_results, title: str):
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    # plot benchmark results
    pal = iter(sns.color_palette("bright", 10))
    import matplotlib.ticker as ticker

    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=300)
    ax2 = ax1.twinx()
    mention_reference_timing = True

    for benchmark in benchmark_results:
        benchmark_details = benchmark_results[benchmark]["benchmark_details"]
        label = f"{benchmark_details.nnp}/{benchmark_details.platform}/"

        ax1.plot(
            benchmark_results[benchmark]["nr_of_atoms"],
            benchmark_results[benchmark]["timing"],
            label=label,
            color=next(pal),
        )
        if benchmark_details.platform == "CUDA":
            ax2.plot(
                benchmark_results[benchmark]["nr_of_atoms"],
                np.array(benchmark_results[benchmark]["gpu_memory_footprint"])
                * 1.024e-9,
                label=f"{label}-gpu mem",
                linestyle="--",
                color=next(pal),
            )
        if (
            "reference" in benchmark_results[benchmark].keys()
            and mention_reference_timing == True
        ):
            mention_reference_timing = False
            ax1.plot(
                benchmark_results[benchmark]["nr_of_atoms"],
                benchmark_results[benchmark]["reference"],
                label=f"MM/{benchmark_details.platform}/reference",
                color=next(pal),
            )

    ax1.set_xlabel("Nr of atoms", fontsize=14)
    ax1.set_ylabel("time (s)", fontsize=14)
    ax2.set_ylabel("memory (Gb)", fontsize=14)
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.grid(None)
    # ask matplotlib for the plotted objects and their labels
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()


def perform_benchmark():
    """
    Perform the benchmark simulations to plot single point energy execution time/GPU memory consumption vs number of atoms for a set of NNP and a reference MM potential.
    """
    from exs.quams.benchmark import Benchmark

    # perform benchmark and save results
    benchmark_results = dict()

    with Progress() as progress:
        task1 = progress.add_task(
            "[green]Performing simulation ...", total=len(system_to_benchmark)
        )

        for bench in system_to_benchmark:
            print("#=================================#")

            benchmark = Benchmark()
            benchmark_name = f"{bench.nnp}_{bench.platform}_{bench.implementation}"
            print(f"{benchmark_name}")

            print("#-----------------------------------#")
            spacing = np.linspace(10, 800, 4)
            benchmark.run_benchmark(
                bench.nnp,
                (WaterCluster(n_waters=int(n_waters)) for n_waters in spacing),
                True,
                platform=bench.platform,
                implementation=bench.implementation,
            )
            benchmark_results[benchmark_name] = {
                "nr_of_atoms": [int(n_waters) * 3 for n_waters in spacing],
                "timing": benchmark.qml_timing,
                "gpu_memory_footprint": benchmark.gpu_memory,
                "reference": benchmark.reference_timing,
                "benchmark_details": bench,
            }
            benchmark = None
            log.info(f"Done with {benchmark_name}")
            print(f"Results: {benchmark_results[benchmark_name]}")
            progress.update(task1, advance=1)

    # plot and save
    plot_timing(benchmark_results, "watercluster benchmark")


if __name__ == "__main__":
    # ------------------------------------------------------#
    # Defining the benchmark system
    # ------------------------------------------------------#
    benchmark_details = namedtuple("BenchmarkSystem", "nnp, platform, implementation")
    system_to_benchmark = [
        benchmark_details(nnp="ani2x", platform="CUDA", implementation="nnpops"),
        benchmark_details(nnp="ani2x", platform="CPU", implementation="torchani"),
        benchmark_details(nnp="ani2x", platform="CPU", implementation="nnpops"),
        benchmark_details(nnp="ani2x", platform="CUDA", implementation="torchani"),
    ]
    # ------------------------------------------------------#
    # ------------------------------------------------------#

    fire.Fire(perform_benchmark)
