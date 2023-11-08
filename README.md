<p align="center">
  <img src="https://github.com/Exscientia/StableNetGuardOwl/assets/31651017/6e72dbdd-3fae-4463-bde3-bbaf54b459a7" alt="StableNetGuardOwl Logo" width="150">
  <h3 align="center">StableNetGuardOwl: Ensuring Stability in Neural Network Potentials</h3>
</p>
<p align="center">

[//]: # (Badges)
![Continuous Integration](https://github.com/Exscientia/StableNetGuardOwl/actions/workflows/CI.yaml/badge.svg?branch=main)
![Code Coverage](https://codecov.io/gh/Exscientia/StableNetGuardOwl/branch/main/graph/badge.svg)
![Supported Python versions](https://img.shields.io/badge/python-%5E3.10-blue.svg)
![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![GitHub Release](https://badgen.net/github/release/Exscientia/StableNetGuardOwl)
![GitHub License](https://img.shields.io/github/license/Exscientia/StableNetGuardOwl?color=green)
</p>

---

# Overview

StableNetGuardOwl provides a robust suite for conducting stability tests on Neural Network Potentials (NNPs) used in molecular simulations. These tests are critical in validating NNPs for research and industrial applications, ensuring accuracy and reliability.

## Features

StableNetGuardOwl supports stability tests for NNPs integrated with `openMM` and those implemented within `openmm-ml` or the Exscientia `physics-ml` package.  
Currently this supports a range of NNPs including but not limited to `SchNET`, `PaiNN`, `MACE`, and `nequip`.

## Test Matrix

The following table provides an overview of the test environments, systems, and properties assessed by StableNetGuardOwl:


| Environment     | Test System                                              | Thermodynamic Ensemble | Test Properties                                                                                    |
|-----------------|----------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------|
| Vacuum          | HiPen set                                                | -                      | Bond/angle deviation, potential energy convergence                                                |
| Vacuum          | Example molecules with various functional groups         | -                      | Bond/angle deviation, potential energy convergence, geometric convergence                        |
| Vacuum          | Dipeptide alanine                                        | -                      | Relaxed 2D torsion scan around phi/psi dihedrals                                                 |
| Vacuum/Water    | Dipeptide alanine                                        | NpT, NVE, NVT          | Bond/angle deviation, potential energy, O-O rdf, density [NpT], energy conservation [NVE], phi/psi distribution |
| Water           | Waterbox                                                 | NpT, NVE, NVT          | Bond/angle deviation, potential energy, O-O rdf, density [NpT], energy conservation [NVE]         |
| Organic Solvent | Butane, Cyclohexane, Ethane, Isobutane, Methanol, Propane | NpT, NVE, NVT         | Potential energy, density, heat of vaporization, heat capacity, compressibility                   |

## Customization

StableNetGuardOwl allows users to customize or extend tests by inheriting from base classes tailored for specific stability assessments.

### Components

- **Stability Testing Script**: Located in the `scripts` directory (`perform_stability_test.py`).
- **Results Visualization Notebook**: Found in the `notebooks` directory (`visualize_results.ipynb`).


Each test outputs:
1. A PDB file for the molecular topology.
2. A CSV file for the properties being monitored.
3. A DCD file for the molecular dynamics trajectory.

## Usage

To initiate a stability test, navigate to the root directory of StableNetGuardOwl and run the following command:

```bash
python scripts/perform_stability_test.py scripts/test_config.yaml
There is an example `test_config.yaml` file provided in the `scripts` directory that provides default parameters for the most common test systems.

### Example
For a stability test using a pure 15 Angstrom waterbox the `config.yaml` file may look like this
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

To visualize the results, use the `visualize_results.ipynb` notebook.

```python
MonitoringPlotter("trajectory.dcd", 
                  "topology.pdb", 
                  "data.csv")
```

### Additional Systems and Protocols
Beyond the primary test environments, StableNetGuardOwl can evaluate small molecules defined by SMILES strings or sourced from datasets such as HiPen. Various protocols are available to investigate molecular dynamics at multiple temperatures, different thermodynamic ensembles, and specific degrees of freedom (DOF) for potential energy scans.


### Protocols
Currently, the following protocols are available:

- MultiTemperatureProtocol. Perform simulations at different temperatures.
- PropagationProtocol. Perform MD simulation in a given thermodynamic ensemble (NpT, NVT, NVE).
- BondProfileProtocol. Stretch bond along its bond axis starting with 0.5 to 5 Angstrom.

## Examples
### Running a DOF Scan
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


