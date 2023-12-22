import os

from typing import Tuple, List, Optional, Dict, Iterator
from loguru import logger as log

available_nnps_and_implementation = [
    ("ani2x", "nnpops"),
    ("ani2x", "torchani"),
]

gh_available_nnps_and_implementation = [
    ("ani2x", "torchani"),
]

gpu_memory_constrained_nnps_and_implementation = [
    ("ani2x", "torchani"),
]


def get_available_nnps_and_implementation() -> list:
    """Return a list of available neural network potentials and implementations"""
    IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
    if IN_GITHUB_ACTIONS:
        return gh_available_nnps_and_implementation
    else:
        return available_nnps_and_implementation


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in the data folder.

    In the source distribution, these files are in ``guardowl/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the folder).

    """

    from pkg_resources import resource_filename

    fn = resource_filename("guardowl", relative_path)

    if not os.path.exists(fn):
        raise ValueError(f"Sorry! {fn} does not exist.")

    return fn


def _logo():
    logo = r"""
           ^...^                    
          / o,o \
          |):::(|
        ====w=w===
          """
    return logo


def _set_loglevel(level="WARNING"):
    from loguru import logger
    import sys

    logger.remove()  # Remove all handlers added so far, including the default one.
    logger.add(sys.stderr, level=level)


def extract_drugbank_tar_gz():
    """
    Extracts a .tar.gz archive to the specified path.
    """
    import tarfile
    import importlib.resources as pkg_resources

    # This assumes 'my_package.data' is the package and 'filename' is the file in the 'data' directory
    with pkg_resources.path("guardowl.data", "drugbank.tar.gz") as DATA_DIR:
        with pkg_resources.path("guardowl.data", "drugbank") as extract_path:
            log.info(f"{DATA_DIR=}")
            log.info(f"{extract_path=}")
            # Check if the extraction is already done
            if not os.path.exists(extract_path):
                with tarfile.open(DATA_DIR, "r:gz") as tar:
                    tar.extractall(path=extract_path)
                    log.debug(f"Extracted {DATA_DIR} to {extract_path}")
            else:
                log.debug(f"Extraction already done")


def _generate_file_list_for_minimization_test(
    dir_name: str = "drugbank", shuffel: bool = False
) -> Dict[str, List]:
    import os

    import importlib.resources as pkg_resources
    import numpy as np

    # This assumes 'my_package.data' is the package and 'filename' is the file in the 'data' directory
    with pkg_resources.path("guardowl.data", f"{dir_name}") as DATA_DIR:
        log.info("Reading in data for minimzation test")
        log.debug(f"Reading from {DATA_DIR}")
        # read in all directories in DATA_DIR
        directories = [x[0] for x in os.walk(DATA_DIR)]
        if shuffel:
            np.random.shuffle(directories)
        # read in all xyz files in directories
        minimized_xyz_files = []
        start_xyz_files = []
        sdf_files = []
        for directory in directories:
            all_files = os.listdir(directory)
            # test if there is a xyz, orca and sdf file in the list and only then continue
            test_orca = any("orca" in file for file in all_files)
            test_xyz = any("xyz" in file for file in all_files)
            test_sdf = any("sdf" in file for file in all_files)
            if (test_orca and test_xyz and test_sdf) == False:
                log.debug(f"Skipping {directory}")
                print(f"Skipping {directory}")
                continue

            for file in all_files:
                if file.endswith(".xyz"):
                    if file.startswith("orca"):
                        minimized_xyz_files.append(os.path.join(directory, file))
                    else:
                        start_xyz_files.append(os.path.join(directory, file))
                        sdf_files.append(
                            os.path.join(directory, file.replace(".xyz", ".sdf"))
                        )

    return {
        "minimized_xyz_files": minimized_xyz_files,
        "unminimized_xyz_files": start_xyz_files,
        "sdf_files": sdf_files,
        "directories": directories,
        "total_number_of_systems": len(minimized_xyz_files),
    }


def _generate_input_for_minimization_test(
    files: Dict[str, List]
) -> Iterator[Tuple[Tuple[str, List], Tuple[str, List]]]:
    # read in coordinates from xyz files
    def read_positions(files) -> Iterator[Tuple[str, List]]:
        for file in files:
            with open(file, "r") as f:
                lines = f.readlines()
                positions = [[float(x) for x in line.split()[1:]] for line in lines[2:]]
                yield file, positions

    minimized_xyz_files = files["minimized_xyz_files"]
    start_xyz_files = files["unminimized_xyz_files"]
    minimized_positions = read_positions(minimized_xyz_files)
    start_positions = read_positions(start_xyz_files)

    return zip(minimized_positions, start_positions)
