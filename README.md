<p align="center">
<img src="https://github.com/Exscientia/StableNetGuardOwl/assets/31651017/6e72dbdd-3fae-4463-bde3-bbaf54b459a7" alt="Simple Icons" width=150>
<h3 align="center">StableNetGuardOwl: Perform stability tests for Neural Network Potentials</h3>
</p>
<p align="center">
  
[//]: # (Badges)
[![CI](https://github.com/Exscientia/StableNetGuardOwl/actions/workflows/CI.yaml/badge.svg?branch=main)](https://github.com/Exscientia/StableNetGuardOwl/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/Exscientia/StableNetGuardOwl/branch/main/graph/badge.svg)](https://codecov.io/gh/Exscientia/StableNetGuardOwl/branch/main)
[![Supported Python versions](https://img.shields.io/badge/python-%5E3.10-blue.svg)](https://docs.python.org/3/whatsnew/index.html)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github release](https://badgen.net/github/release/Exscientia/StableNetGuardOwl)](https://github.com/Exscientia/StableNetGuardOwl/)
[![GitHub license](https://img.shields.io/github/license/Exscientia/StableNetGuardOwl?color=green)](https://github.com/Exscientia/StableNetGuardOwl/blob/main/LICENSE)
</p>

---

# What this package contains

## Stability testing

This repository provides the essential code for performing various stability tests with Neural Network Potentials (NNPs). Tests are currently limited to `openMM` and the Neural Network Potentials implemented in `openmm-ml`. The `physics-ml` package of Exscientia provides models trained on the `SPICE` dataset including implementation in `openmm-ml` for the following NNPs: `SchNET`, `PaiNN`, `MACE` and `nequip`.

The tests currently focus on small molecules in vacuum and bulk properties of water.

| Environment | Test system | Thermodynamic ensemble | Test property |
| --- | ----------- | ----------- | ----------- |
| Vacuum | HiPen set | - | Bond/angle deviation, potential energy convergence |
| Vacuum | Example molecules for relevant functional groups | - | Bond/angle deviation, potential energy convergence |
| Vacuum | Dipeptide alanine |  | relaxed 2D torsion scan around phi/psi dihedral |
| Vacuum/Water | Dipeptide alanine | NpT, NVE, NVT | Bond/angle deviation, potential energy convergence, O-O rdf, density [NpT], energy conservation [NVE], phi/psi distribution |
| Water | Waterbox | NpT, NVE, NVT | Bond/angle deviation, potential energy convergence, O-O rdf, density [NpT], energy conservation [NVE] |
| Organic solvent | n-octanol, cyclohexane | NpT | potential energy convergence, density |




The tests and systems can be customized or redefined by inheriting from the appropriate base classes. Key components include:

- Script for stability testing: located in the scripts directory (`perform_guardowls.py``).
- Results visualization notebook: found in the notebooks directory (`visualize_guardowls.ipynb``).

Each stability test produces three types of output files:

1. A PDB file defining the molecular system's topology.
2. A CSV file containing the monitored properties.
3. A DCD trajectory file to visualize the system's temporal evolution.

To perform a stability test the general syntax is as follows:
Command Syntax
```bash
python scripts/perform_guardowls.py -c config.yaml
```
There is an example `config.yaml` file provided in the `scripts` directory that provides default parameters for the most common test systems.

### Example
For a stability test using a pure 15 Angstrom waterbox the config.yaml file looks like shown below
```
tests:
  - protocol: "waterbox_protocol"  # which protocol is performed
    edge_length: 15                # waterbox edge length in Angstrom
    ensemble: "NVT"                # thermodynamic esamble that is used. Oter options are 'NpT' and 'NVE'.
    nnp: "ani2x"                   # the NNP used
    implementation: "nnpops"       # the implementation if multiple are available
    annealing: false               # simulated annealing to slowly reheat the system at the beginning of a simulation
    nr_of_simulation_steps: 10_000 # number of simulation steps
    temperature: 300               # in Kelvin
```
It defines the potential (nnp and implementation), the number of simulation steps, temperature in Kelvin, and edge length of the waterbox in Angstrom as well as the thermodynamic ensemble (`NVT`). Passing this to the `perform_guardowls.py` script runs the tests

To visualize the results, use the visualize_guardowls.ipynb notebook.

```python
MonitoringPlotter("trajectory.dcd", 
                  "topology.pdb", 
                  "data.csv")
```

### Other Test Systems
Test systems that can be used for different protocols include small molecules in vacuum (either defined with SMILES string or taken from the HiPen dataset), alanine-dipeptide in vacuum and solution, pure waterbox and degree of freedom potential energy scan, e.g., along a bond, angle, or torsion angle.

### Protocols
Currently, the following protocols are available:

- MultiTemperatureProtocol. Perform simulations at different temperatures.
- PropagationProtocol. Perform MD simulation in a given thermodynamic ensemble (NpT, NVT, NVE).
- BondProfileProtocol. Stretch bond along its bond axis starting with 0.5 to 5 Angstrom.

### Examples
To perform a DOF scan over a bond in ethanol you need to generate a yaml file containing the following (scan over bond connecting atom index 0 and 2. 

```yaml
tests:
  - protocol: "perform_DOF_scan"
    nnp: "ani2x"
    implementation: "torchani"
    DOF_definition: { "bond": [0, 2] }
    molecule_name: "ethanol"
```


### Copyright

Copyright (c) 2023, Marcus Wieder & QuaMS product team @ Exscientia


