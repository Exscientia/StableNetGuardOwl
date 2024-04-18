# ------------------ IMPORTS ------------------#
from loguru import logger as log
from openmm import Platform
from typing import Any, Dict

# ------------------ IMPORTS ------------------#


def get_fastest_platform() -> Platform:
    """
    Identifies and returns the fastest available OpenMM platform.

    Returns
    -------
    platform : Platform
        The fastest available OpenMM platform object.
    """

    from openmmtools.utils import get_fastest_platform

    platform = get_fastest_platform()
    log.info(f"Using platform {platform.getName()}")
    return platform


def setup_logging_and_output() -> str:
    """
    Sets up the logging and output directory.

    Returns
    -------
    output_folder : str
        The path to the output directory.
    """

    from pathlib import Path

    output_folder = "test_stability_protocol"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    return output_folder


def validate_input(nnp: str):
    """
    Validates the combination of neural network potential.

    Parameters
    ----------
    nnp : str
        The neural network potential to validate.

    Raises
    ------
    RuntimeError
        If the NNP is invalid.
    """

    from guardowl.utils import available_nnps

    if nnp not in available_nnps:
        error_message = f"Invalid nnp. Got {nnp}"
        log.error(error_message)
        raise RuntimeError(error_message)


from openmm.app import StateDataReporter


def create_state_data_reporter() -> StateDataReporter:
    """
    Creates and returns a StateDataReporter object for OpenMM simulations.

    Returns
    -------
    reporter : StateDataReporter
        An instance of OpenMM's StateDataReporter.
    """

    return StateDataReporter(
        file=None,
        reportInterval=500,
        step=True,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        volume=True,
        speed=True,
    )


def load_config(config_file_path: str) -> Dict[str, Any]:
    """
    Loads and returns the configuration from a YAML file.

    Parameters
    ----------
    config_file_path : str
        The path to the YAML configuration file.

    Returns
    -------
    config : Dict[str, Any]
        The loaded configuration as a dictionary.
    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """

    import yaml

    try:
        with open(config_file_path, "r") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        log.error(f"Configuration file not found: {config_file_path}")
        raise
    except yaml.YAMLError as exc:
        log.error(f"Error parsing YAML configuration: {exc}")
        raise


from guardowl.testsystems import LiquidOption


def process_test(test: Dict[str, Any], platform: Platform, output: str) -> None:
    """
    Processes a single test configuration.

    Parameters
    ----------
    test : Dict[str, Any]
        The test configuration.
    platform : Platform
        The OpenMM platform to use for the test.
    output : str
        The output directory for the test results.
    """
    from guardowl.protocols import (
        run_DOF_scan,
        run_small_molecule_test,
        run_waterbox_test,
        run_alanine_dipeptide_test,
        run_organic_liquid_test,
    )

    protocol_function = {
        "small_molecule_test": run_small_molecule_test,
        "waterbox_test": run_waterbox_test,
        "alanine_dipeptide_test": run_alanine_dipeptide_test,
        "organic_liquid_test": run_organic_liquid_test,
        "DOF_scan": run_DOF_scan,
    }.get(test.get("protocol"))

    if protocol_function is None:
        log.error(f"Unknown protocol: {test.get('protocol')}")
        raise NotImplementedError(f"Unknown protocol: {test.get('protocol')}")
    else:
        protocol_function(**{k: test[k] for k in test if k != "protocol"})


def main(config: str) -> None:
    """
    Main function to run stability tests based on a provided configuration.

    Parameters
    ----------
    config_path : str
        The path to the YAML configuration file.
    """
    from guardowl.setup import PotentialFactory

    log.info(f"Loaded config from {config}")
    config = load_config(config)
    platform = get_fastest_platform()
    output = setup_logging_and_output()

    for potential in config.get("potentials", []):

        for test in config.get("tests", []):
            print("--------- Test starts --------- ")
            test["output_folder"] = output
            test["reporter"] = create_state_data_reporter()
            test["platform"] = platform

            test["nnp"] = PotentialFactory().initialize_potential(potential)
            test["output_folder"] = (
                f"{test['output_folder']}/{potential['provider']}_{potential['model_name']}"
            )
            process_test(test, platform, output)
            print("--------- Test finishs --------- ")


def _setup_logging():
    import logging
    from guardowl.utils import _logo, _set_loglevel

    print(_logo())

    logging.getLogger().setLevel(logging.CRITICAL)

    _set_loglevel("INFO")


if __name__ == "__main__":
    import typer
    import warnings
    import torch

    torch._C._jit_set_nvfuser_enabled(False)

    _setup_logging()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        typer.run(main)
