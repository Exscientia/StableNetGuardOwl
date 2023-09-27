# Perform stability tests for Neural Network Potentials

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/Exscientia/stability_test/workflows/CI/badge.svg)](https://github.com/Exscientia/stability_test/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/Exscientia/stability_test/branch/main/graph/badge.svg)](https://codecov.io/gh/Exscientia/stability_test/branch/main)
[![Supported Python versions](https://img.shields.io/badge/python-%5E3.10-blue.svg)](https://docs.python.org/3/whatsnew/index.html)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github release](https://badgen.net/github/release/Exscientia/stability_test)](https://github.com/Exscientia/stability_test/)
[![GitHub license](https://img.shields.io/github/license/Exscientia/stability_test?color=green)](https://github.com/Exscientia/stability_test/blob/main/LICENSE)


---

# What this package contains

## Stability testing

This repository provides the essential code for performing a variety of stability tests on test systems. The tests and systems can be customized or redefined by inheriting from the appropriate base classes. Key components include:

- Script for Stability Testing: Located in the scripts directory (perform_stability_tests.py).
- Results Visualization Notebook: Found in the notebooks directory (visualize_stability_tests.ipynb).

Each stability test produces three types of output files:

1. A PDB file defining the molecular system's topology.
2. A CSV file containing the monitored properties.
3. A DCD trajectory file to visualize the system's temporal evolution.

Command Syntax
'''bash
python perform_stability_tests.py TESTSYSTEM OPTIONS
'''
Different TESTSYSTEM options provide varying control parameters.

### Example
For a stability test using a pure waterbox:


'''bash
python perform_stability_tests.py waterbox 20 NpT ani2x torchani --nr_of_simulation_steps 1000
'''
This runs a NpT simulation in a 20-angstrom waterbox using the ani-2x potential and its torchani implementation.

To visualize the results, use the visualize_stability_tests.ipynb notebook.

'''python
MonitoringPlotter("trajectory.dcd", 
                  "topology.pdb", 
                  "data.csv")
'''

### Other Test Systems
Options for TESTSYSTEM include vacuum, alanine-dipeptide, and DOF (Degree Of Freedom scan, e.g., bond, angle, or torsion scan).

### Protocols
Currently, the following protocols are available:

MultiTemperatureProtocol
PropagationProtocol
BondProfileProtocol


###Examples
DOF Scan Over a Bond in Ethanol:

'''bash
python perform_stability_tests.py DOF ani2x torchani "{'bond' : [0, 2]}" ethanol
'''

### Copyright

Copyright (c) 2023, Marcus Wieder & QuaMS product team @ Exscientia


