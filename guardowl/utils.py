import os

available_nnps_and_implementation = [
    ("ani2x", "nnpops"),
    ("ani2x", "torchani"),
]

gh_available_nnps_and_implementation = [
    ("ani2x", "torchani"),
]


def get_available_nnps_and_implementation() -> list:
    """Return a list of available neural network potentials and implementations"""
    IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
    if IN_GITHUB_ACTIONS:
        return gh_available_nnps_and_implementation
    else:
        return available_nnps_and_implementation


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
