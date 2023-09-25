import subprocess


def generate_args(script: str):
    args = {
        "hipen_vacuum_args": [  # Check if script can be executed with the correct arguments for a HiPen molecule
            "python",
            script,
            "vacuum",
            "1",
            "ani2x",
            "nnpops",
            "50",
        ],
        "waterbox_args": [  # Check if script can be executed with the correct arguments for a waterbox
            "python",
            script,
            "waterbox",
            "20",
            "NpT",
            "ani2x",
            "torchani",
            "False",
            "10",
        ],
        "alanine_dipeptide_solvent_args": [  # Check if script can be executed with the correct arguments for a alanine dipeptide in solvent
            "python",
            script,
            "alanine-dipeptide",
            "solvent",
            "ani2x",
            "torchani",
            "NpT",
            "10",
        ],
        "alanine_dipeptide_vacuum_args": [  # Check if script can be executed with the correct arguments for a alanine dipeptide in vacuum
            "python",
            script,
            "alanine-dipeptide",
            "vacuum",
            "ani2x",
            "torchani",
            "",
            "10",
        ],
        "DOF_ethane_args": [  # Check if script can be executed with the correct arguments for a alanine dipeptide in vacuum        "python",
            script,
            "DOF",
            "ani2x",
            "torchani",
            "{'bond' : [0, 2]}",
            "ethanol",
        ],
    }
    return args


def test_script_execution_vacuum(
    script: str = "scripts/perform_stability_tests.py",
) -> None:
    print(f"Testing {script}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script, "--help"], capture_output=True)
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0

    args = generate_args(script)["hipen_vacuum_args"]
    ret = subprocess.run(
        args,
        capture_output=True,
    )
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0


def test_script_execution_waterbox(
    script: str = "scripts/perform_stability_tests.py",
) -> None:
    print(f"Testing {script}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script, "--help"], capture_output=True)
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0

    args = generate_args(script)["waterbox_args"]

    ret = subprocess.run(
        args,
        capture_output=True,
    )


def test_script_execution_dipeptide_solvent(
    script: str = "scripts/perform_stability_tests.py",
) -> None:
    print(f"Testing {script}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script, "--help"], capture_output=True)
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0

    args = generate_args(script)["alanine_dipeptide_solvent_args"]

    ret = subprocess.run(
        args,
        capture_output=True,
    )


def test_script_execution_dipeptide_vacuum(
    script: str = "scripts/perform_stability_tests.py",
) -> None:
    print(f"Testing {script}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script, "--help"], capture_output=True)
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0

    args = generate_args(script)["alanine_dipeptide_vacuum_args"]

    ret = subprocess.run(
        args,
        capture_output=True,
    )
