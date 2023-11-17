import subprocess
import pytest

import os

TEST_TO_PERFORM = [
    "stability_test_hipen.yaml",
    "stability_test_waterbox.yaml",
    "stability_test_alanine_dipeptide.yaml",
    "stability_test_pure_liquid.yaml",
]


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

if IN_GITHUB_ACTIONS:
    # exclude alanine dipeptide test
    TEST_TO_PERFORM = TEST_TO_PERFORM[:-1]


@pytest.mark.parametrize("config_file_path", TEST_TO_PERFORM)
@pytest.mark.parametrize("script_file_path", ["scripts/perform_stability_tests.py"])
def test_script_execution(config_file_path: str, script_file_path: str) -> None:
    print(f"Testing {script_file_path}")
    print(f"Using {config_file_path}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script_file_path, "--help"], capture_output=True)
    print("Output from --help:")
    print(ret.stdout.decode("utf-8"))
    print("Error from --help:")
    print(ret.stderr.decode("utf-8"))
    assert ret.returncode == 0

    # Update the arguments to match your argparse in the script
    args = f"python {script_file_path} {config_file_path}".split()
    ret = subprocess.run(args, capture_output=True)
    print("Script Output:")
    print(ret.stdout.decode("utf-8"))

    print("Script Error:")
    print(ret.stderr.decode("utf-8"))

    assert ret.returncode == 0
