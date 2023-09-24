import subprocess


def test_stability_tests_script(
    script: str = "scripts/perform_stability_tests.py",
) -> None:
    print(f"Testing {script}")
    # Check if script exists and can be executed
    ret = subprocess.run(["python", script, "--help"], capture_output=True)
    print(ret.stdout.decode("utf-8"))
    assert ret.returncode == 0

    # Check if script can be executed with the correct arguments for a HiPen molecule
    hipen_vacuum_args = [
        "python",
        script,
        "vacuum",
        "1",
        "ani2x",
        "nnpops",
        "50",
    ]
    # Check if script can be executed with the correct arguments for a waterbox

    waterbox_args = [
        "python",
        script,
        "waterbox",
        "20",
        "NpT",
        "ani2x",
        "torchani",
        "False",
        "10",
    ]
    # Check if script can be executed with the correct arguments for a alanine dipeptide in solvent
    alanine_dipeptide_solvent_args = [
        "python",
        script,
        "alanine-dipeptide",
        "solvent",
        "ani2x",
        "torchani",
        "NpT",
        "10",
    ]
    # Check if script can be executed with the correct arguments for a alanine dipeptide in vacuum
    alanine_dipeptide_vacuum_args = [
        "python",
        script,
        "alanine-dipeptide",
        "vacuum",
        "ani2x",
        "torchani",
        "",
        "10",
    ]
    # Check if script can be executed with the correct arguments for a alanine dipeptide in vacuum
    DOF_ethane_args = [
        "python",
        script,
        "DOF",
        "ani2x",
        "torchani",
        "{'bond' : [0, 2]}",
        "ethanol",
    ]

    for args in [
        hipen_vacuum_args,
        # waterbox_args,
        # alanine_dipeptide_solvent_args,
        alanine_dipeptide_vacuum_args,
        DOF_ethane_args,
    ]:
        print(ret.stdout.decode("utf-8"))
        assert ret.returncode == 0

        ret = subprocess.run(
            args,
            capture_output=True,
        )
