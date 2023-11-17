# ------------------ IMPORTS ------------------#
from loguru import logger as log

# ------------------ IMPORTS ------------------#


def get_fastest_platform():
    from openmmtools.utils import get_fastest_platform

    platform = get_fastest_platform()
    log.info(f"Using platform {platform.getName()}")
    return platform


def setup_logging_and_output():
    from pathlib import Path

    output_folder = "test_stability_protocol"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    return output_folder


def create_test_system(testsystem_factory, *args, **kwargs):
    return testsystem_factory().generate_testsystems(*args, **kwargs)


def create_state_data_reporter():
    from openmm.app import StateDataReporter

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


def load_config(config_file_path):
    import yaml

    with open(config_file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def main(config: str):
    from guardowl.protocols import (
        run_DOF_scan,
        run_hipen_protocol,
        run_waterbox_protocol,
        run_alanine_dipeptide_protocol,
        run_pure_liquid_protocol,
    )

    log.info(f"Loaded config: {config}")
    config = load_config(config)

    for test in config.get("tests", []):
        protocol = test.get("protocol")
        
        # add reporter and output folder
        test["output_folder"] = setup_logging_and_output()
        test["reporter"] = create_state_data_reporter()
        test["platform"] = get_fastest_platform()

        if protocol == "hipen_protocol":
            log.info("Performing hipen protocol")
            run_hipen_protocol(**{k: test[k] for k in test if k != "protocol"})

        elif protocol == "waterbox_protocol":
            log.info("Performing waterbox protocol")
            run_waterbox_protocol(**{k: test[k] for k in test if k != "protocol"})

        elif protocol == "alanine_dipeptide_protocol":
            log.info("Performing alanine dipeptide protocol")
            run_alanine_dipeptide_protocol(
                **{k: test[k] for k in test if k != "protocol"}
            )
        elif protocol == "pure_liquid_protocol":
            log.info("Performing pure liquid protocol")
            run_pure_liquid_protocol(**{k: test[k] for k in test if k != "protocol"})

        elif protocol == "DOF_scan":
            log.info("Performing DOF protocol")
            run_DOF_scan(**{k: test[k] for k in test if k != "protocol"})

        else:
            log.warning(f"Unknown protocol: {protocol}")
            raise NotImplementedError(f"Unknown protocol: {protocol}")


def _setup_logging():
    import logging
    from guardowl.utils import _logo, _set_loglevel

    print(_logo())

    logging.getLogger().setLevel(logging.CRITICAL)

    _set_loglevel("INFO")


if __name__ == "__main__":
    import typer
    import warnings

    _setup_logging()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        typer.run(main)
