import os

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
