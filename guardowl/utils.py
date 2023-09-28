import os

available_nnps_and_implementation = [
    ("ani2x", "nnpops"),
    ("ani2x", "torchani"),
    ("ani1ccx", "nnpops"),
    ("ani1ccx", "torchani"),
]

gh_available_nnps_and_implementation = [
    ("ani2x", "torchani"),
    ("ani1ccx", "torchani"),
]


def get_available_nnps_and_implementation() -> list:
    """Return a list of available neural network potentials and implementations"""
    IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
    if IN_GITHUB_ACTIONS:
        return gh_available_nnps_and_implementation
    else:
        return available_nnps_and_implementation
