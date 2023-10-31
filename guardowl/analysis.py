import mdtraj as md
import numpy as np
from typing import List, Tuple


class PropertyCalculator:
    """
    A class for calculating various properties for a molecular dynamics trajectory.

    Attributes
    ----------
    md_traj : md.Trajectory
        The molecular dynamics trajectory.

    Methods
    -------
    calculate_water_rdf()
        Calculate the radial distribution function of water molecules.
    monitor_water_bond_length()
        Monitor the bond length between water molecules.
    monitor_water_angle()
        Monitor the angle between water molecules.
    monitor_bond_length(bond_pairs)
        Monitor the bond length for specific atom pairs.
    monitor_angle_length(angle_list)
        Monitor the angles for specific sets of atoms.

    """

    def __init__(self, md_traj: md.Trajectory) -> None:
        """
        Initialize the PropertyCalculator with a molecular dynamics trajectory.

        Parameters
        ----------
        md_traj : md.Trajectory
            The molecular dynamics trajectory to be analyzed.

        """
        self.md_traj = md_traj

    def calculate_heat_capacity(
        self, total_energy: np.array, volumn: np.array
    ) -> float:
        """
        Calculate the heat capacity of the trajectory.
        C_p = <\Delta E^2> / k_B T^2 V
        """
        from openmm.unit import kelvin
        from .constants import kB

        mean_energy = np.mean(total_energy)
        mean_volume = np.mean(volumn)

        # Calculate the mean square fluctuation of the energy
        mean_square_fluctuation_energy = np.mean((total_energy - mean_energy) ** 2)

        # Calculate Cp using the formula
        Cp = mean_square_fluctuation_energy / (kB * (300 * kelvin) ** 2 * mean_volume)

        return Cp

    def calculate_isothermal_compressability_kappa_T(self):
        from .constants import temperature

        return md.isothermal_compressability_kappa_T(self.md_traj, temperature)

    def calculate_thermal_expansion_alpha_P(self, pot_energy: np.array):
        from .constants import temperature

        return md.thermal_expansion_alpha_P(self.md_traj, temperature, pot_energy)

    def calculate_water_rdf(self):  # type: ignore
        """
        Calculate the radial distribution function (RDF) for water molecules in the trajectory.

        Returns
        -------
        np.ndarray
            The RDF values for the water molecules.
        """
        oxygen_pairs = self.md_traj.top.select_pairs(
            "name O and water", "name O and water"
        )
        bins = 300
        r_max = 1
        r_min = 0.01

        mdtraj_rdf = md.compute_rdf(
            self.md_traj, oxygen_pairs, (r_min, r_max), n_bins=bins
        )

        return mdtraj_rdf

    def _extract_water_bonds(self) -> List[Tuple[int, int]]:
        bond_list = []
        for bond in self.md_traj.topology.bonds:
            if bond.atom1.residue.name == "HOH" and bond.atom2.residue.name == "HOH":
                bond_list.append((bond.atom1.index, bond.atom2.index))
        return bond_list

    def _extract_bonds_except_water(self) -> List[Tuple[int, int]]:
        bond_list = []
        for bond in self.md_traj.topology.bonds:
            if bond.atom1.residue.name != "HOH" and bond.atom2.residue.name != "HOH":
                bond_list.append((bond.atom1.index, bond.atom2.index))
        return bond_list

    def monitor_water_bond_length(self):  # type: ignore
        """
        Monitor the bond length between water molecules in the trajectory.

        Returns
        -------
        np.ndarray
            The bond lengths between water molecules.

        """

        bond_list = self._extract_water_bonds()
        return self.monitor_bond_length(bond_list)

    def monitor_bond_length_except_water(self):  # type: ignore
        bond_list = self._extract_bonds_except_water()
        bond_length = self.monitor_bond_length(bond_list)
        compare_to = bond_length[0]
        bond_diff = np.abs(bond_length - compare_to)
        return bond_diff

    def monitor_water_angle(self):  # type: ignore
        """
        Monitor the angle between water molecules in the trajectory.

        Returns
        -------
        np.ndarray
            The angles between water molecules.

        """

        def _extract_angles() -> list:
            """
            Helper function to extract angles between water molecules.

            Returns
            -------
            List[List[int]]
                A list of atom index triplets representing the angles to monitor.

            """

            angle_list = []
            for bond_1 in self.md_traj.top.bonds:
                # skip if bond is not a water molecule
                if bond_1.atom1.residue.name != "HOH":
                    continue
                for bond_2 in self.md_traj.top.bonds:
                    # skip if bond is not a water molecule
                    if bond_2.atom1.residue.name != "HOH":
                        continue
                    water = {}
                    for bond in [bond_1, bond_2]:
                        water[bond.atom1.index] = bond.atom1
                        water[bond.atom2.index] = bond.atom2
                    # skip if atoms are not part of the same molecule
                    if len(water.keys()) != 3:
                        continue

                    sorted_water = [
                        water[key].index for key in sorted(water.keys(), reverse=True)
                    ]  # oxygen is first
                    angle_list.append(
                        [sorted_water[1], sorted_water[2], sorted_water[0]]
                    )

            return [list(x) for x in set(tuple(x) for x in angle_list)]

        angle_list = _extract_angles()
        return self.monitor_angle_length(angle_list)

    def monitor_bond_length(self, bond_pairs: list):  # type: ignore
        """
        Monitor the bond length between specific atom pairs in the trajectory.

        Parameters
        ----------
        bond_pairs : List[Tuple[int, int]]
            A list of atom index pairs whose bond lengths are to be monitored.

        Returns
        -------
        np.ndarray
            The bond lengths for the specified atom pairs.

        """
        bond_length = md.compute_distances(self.md_traj, bond_pairs)
        return bond_length

    def monitor_angle_length(self, angle_list: list):  # type: ignore
        """
        Monitor the angles for specific sets of atoms in the trajectory.

        Parameters
        ----------
        angle_list : List[List[int]]
            A list of atom index triplets whose angles are to be monitored.

        Returns
        -------
        np.ndarray
            The angles for the specified sets of atoms.

        """
        angles = md.compute_angles(self.md_traj, angle_list) * (180 / np.pi)
        return angles

    def monitor_phi_psi(self) -> Tuple[np.ndarray, np.ndarray]:
        phi = md.compute_phi(self.md_traj)[1]
        psi = md.compute_psi(self.md_traj)[1]
        return (phi, psi)
