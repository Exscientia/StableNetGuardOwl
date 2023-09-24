# quams

[![Coverage](https://bitbucket.org/exscientia/quams/downloads/coverage.svg)](https://coverage.readthedocs.io)
[![Supported Python versions](https://img.shields.io/badge/python-%5E3.10-blue.svg)](https://docs.python.org/3/whatsnew/index.html)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![cookiecutter: RSE](https://img.shields.io/badge/cookiecutter-RSE-green?logo=cookiecutter&logoColor=white)](https://bitbucket.org/exscientia/cookiecutter-python-rse)

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/stability_test/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/stability_test/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/stability_test/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/stability_test/branch/main)

---

Perform stability tests for Neural Network Potentials


# What this package contains

## Stability testing

This repo contains the necessary code to perform a set of stability tests on a test system. The stability test and the test system can bue customized or redefined by inheriting from the appropriate base classes. The script that is used to perform a stability test is located in the `scripts` directory: `perform_stability_tests.py` and the notebook used to visualize the results is located in the `notebooks` directory: `visualize_stability_tests.ipynb`. It uses `openmm-ml` and `openmm` to use a trained NNP to perform the simulations.

Every stability test produces three output files: a pdb file that defines the topology of the molecular system, a csv file that contains the monitored properties and a dcd trajectory file that is used to visualize the time evolution of the molecular system.

The `visualize_stability_tests.ipynb` notebook takes all three files as input and visualizes the monitored properties next to the time evolution of the system.

The `perform_stability_tests.py` script takes options with the following syntax: `python perform_stability_tests.py TESTSYSTEM OPTIONS`.
Depeding on the `TESTSYSTEM` different control parameters are made available.

An example for a stability test using a pure waterbox takes the form `python perform_stability_tests.py waterbox EDGE_LENGTH ENSEMBLE NNP IMPLEMENTATION <flag>`, e.g.:

```
python perform_stability_tests.py waterbox 20 NpT ani2x torchani --nr_of_simulation_steps 1000
```
which performs a NpT simulation in a 20 Angstrom waterbox using the `ani-2x` potential and its `torchani` implementation.
To visualize the results use the `visualize_stability_tests.ipynb` notebook. Results are written in `scripts/test_stability_protocol/` and what needs to be update are the three file paths that are passed to the `MonitoringPlotter` instance.

```
MonitoringPlotter("trajectory.dcd", 
                  "topology.pdb", 
                  "data.csv")
```

Other options for `TESTSYSTEM` are `vacuum`, `alanine-dipeptide` or `DOF` (the last defines a degree of freedome scan, i.e. a bond, angle or torsion scan).

Currently there are three protocols available:
- MultiTemperatureProtocol
- PropagationProtocol
- BondProfileProtocol

The MultiTemperatureProtocol performs simulations at 300, 600 and 1200 Kelvin. The PropagationProtocol performs a pure MD simulation and the BondProfileProtocol scans a selected bond to generate an energy profile.

Available `TESTSYSTEM` for out-of-the-box testing are 
- WaterBox
- pure waterbox with varying sizes
- AlanineDipeptideVacuum
- AlanineDipeptideExplicit
- SrcExplicit (medium sized protein in explicit water)
- HipenSystemVacuum (all molecules defined in the HiPen test data used to test free energy convergence in vacuum)
- SmallMoleculeVacuum (ethanol, methanol, ethane, butane, propanol, etc. )

For documentation on the different arguments call `python perform_stability_tests.py waterbox --help`.

## Examples
#### Perform DOF scan over a bond in ethanol

`python perform_stability_tests.py DOF ani2x torchani "{'bond' : [0, 2]}" ethanol`.

### Perform simulation of a HIPEN molecule at different temperatures using ani2x and the nnpops implementation in vacuum

`python perform_stability_tests.py vacuum 1 ani2x nnpops 50000`

### Perform simulation of a 20 A waterbox in the NpT ensemble with simulated annealing phase for 50_000 time steps

`python perform_stability_tests.py waterbox 20 NpT ani2x nnpops True 50000`


## Free energy methdos

There are two free energy methods availalbe in the `quams` package. The theory is outlined here: [absolute](https://exscientia.atlassian.net/wiki/spaces/QuaMS/pages/2623473010/ML+ASFE) and [relative](https://exscientia.atlassian.net/wiki/spaces/QuaMS/pages/2624160101/ML+RSFE) free energies.

### Absolute solvation free energy

Absolute free energies can be calculated using the `absolute_free_energy.py` script.
It takes a SMILES string a  input and performs sequential staged decoupling of the ligand with its surrounding water environment.
This method is in theory applicable to any given environment, but is implemented only for solvation free energies at that point.

### Relative free energy

Relative free energies are implemented in vacuum

### Copyright

Copyright (c) 2023, Marcus Wieder & QuaMS product team @ Exscientia


