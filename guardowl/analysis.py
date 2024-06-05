import mdtraj as md
import numpy as np
from typing import List, Tuple
from loguru import logger as log
from pathlib import Path


class PropertyCalculator:

    def __init__(self, md_traj: md.Trajectory) -> None:
        """
        Calculates various properties for a molecular dynamics trajectory.

        Parameters
        ----------
        md_traj : md.Trajectory
            The molecular dynamics trajectory for analysis.
        """
        self.md_traj = md_traj

    def calculate_heat_capacity(
        self, total_energy: np.array, volumn: np.array
    ) -> float:
        """
        Calculates the heat capacity of the system using the trajectory data.

        Parameters
        ----------
        total_energy : np.array
            The total energy for each frame of the trajectory.
        volume : np.array
            The volume for each frame of the trajectory.

        Returns
        -------
        float
            The calculated heat capacity of the system.

        Notes
        -----
        The formula used for calculation is C_p = <\Delta E^2> / (kB * T^2 * V).
        """
        from .constants import kB, temperature

        mean_energy = np.mean(total_energy)
        mean_volume = np.mean(volumn)

        # Calculate the mean square fluctuation of the energy
        mean_square_fluctuation_energy = np.mean((total_energy - mean_energy) ** 2)

        # Calculate Cp using the formula
        Cp = mean_square_fluctuation_energy / (kB * temperature**2 * mean_volume)
        log.debug(f"heat capacity: {Cp}")
        return Cp

    def calculate_isothermal_compressability_kappa_T(self):
        from .constants import temperature
        from openmm.unit import kelvin

        kappa_T = md.isothermal_compressability_kappa_T(
            self.md_traj, temperature.value_in_unit(kelvin)
        )
        log.debug(f"isothermal_compressability_kappa_T: {kappa_T}")
        return kappa_T

    def calculate_water_rdf(self) -> np.ndarray:
        """
        Calculates the radial distribution function (RDF) for water molecules within the trajectory.

        Returns
        -------
        np.ndarray
            The RDF values for water molecules.
        """
        oxygen_pairs = self.md_traj.topology.select_pairs(
            "name O and water", "name O and water"
        )
        r_max = 1.0
        r_min = 0.01
        bins = 300

        rdf_result = md.compute_rdf(
            self.md_traj,
            pairs=oxygen_pairs,
            r_range=(r_min, r_max),
            bin_width=(r_max - r_min) / bins,
        )
        return rdf_result

    def experimental_water_rdf(self) -> np.ndarray:
        """
        Returns the data for the experimental radial distribution function (RDF) for
        water molecules. This is taken from the file experimental_water_rdf.txt

        Returns
        -------
        np.ndarray
            The RDF values for water molecules.
        """
        # get cwd
        base_path = Path(__file__).parent
        exp_rdf_path = (base_path / "data/experimental_water_rdf.txt").resolve()

        # load experimental water rdf data
        rdf_data = np.loadtxt(exp_rdf_path)

        # convert A to nm for use with mdtraj
        rdf_x = [pt / 10 for pt in rdf_data[:, [0]]]

        rdf_y = rdf_data[:, [1]]

        # return O-O data
        return rdf_x, rdf_y

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

    def monitor_water_bond_length(self) -> np.ndarray:
        """
        Monitors the bond length between water molecules throughout the trajectory.

        Returns
        -------
        np.ndarray
            The bond lengths between water molecules across all frames of the trajectory.
        """
        bond_list = self._extract_water_bonds()
        return self.monitor_bond_length(bond_list)

    def monitor_bond_length_except_water(self):
        """
        This method monitors the bond deviation in molecules that are *not* water molecules throughout the trajectory.
        """
        bond_list = self._extract_bonds_except_water()
        bond_length = self.monitor_bond_length(bond_list)
        compare_to = bond_length[0]
        bond_diff = np.abs(bond_length - compare_to)
        return bond_diff

    def monitor_water_angle(self) -> np.ndarray:
        """
        Monitors the angle between water molecules throughout the trajectory.

        Returns
        -------
        np.ndarray
            The angles between water molecules across all frames of the trajectory.
        """
        angle_list = self._extract_water_angles()
        return self.monitor_angle_length(angle_list)

    def _extract_water_angles(self) -> List[List[int]]:
        """
        Extracts sets of atom indices to compute angles within water molecules.

        Returns
        -------
        List[List[int]]
            A list of lists, where each inner list contains indices of three atoms forming an angle.
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
                angle_list.append([sorted_water[1], sorted_water[2], sorted_water[0]])

        return [list(x) for x in set(tuple(x) for x in angle_list)]

    def monitor_bond_length(self, bond_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Monitors the bond length for specified pairs of atoms across the trajectory

        Parameters
        ----------
        bond_pairs : List[Tuple[int, int]]
            A list of atom index pairs whose bond lengths are to be monitored.

        Returns
        -------
        np.ndarray
            An array of bond lengths for the specified atom pairs across all frames of the trajectory.

        """
        bond_lengths = md.compute_distances(self.md_traj, atom_pairs=bond_pairs)
        return bond_lengths

    def monitor_angle_length(self, angle_list: List[List[int]]) -> np.ndarray:
        """
        Monitors the angles for specified sets of atoms throughout the trajectory.

        Parameters
        ----------
        angle_list : List[List[int]]
            A list of lists, where each inner list contains indices of three atoms forming an angle to be monitored.

        Returns
        -------
        np.ndarray
            An array of angles in degrees for the specified sets of atoms across all frames of the trajectory.

        """
        angles = md.compute_angles(self.md_traj, angle_indices=angle_list) * (
            180 / np.pi
        )
        return angles

    def monitor_phi_psi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monitors the phi and psi dihedral angles throughout the trajectory.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two arrays representing the phi and psi angles in radians for each frame of the trajectory.

        """
        _, phi_angles = md.compute_phi(self.md_traj)
        _, psi_angles = md.compute_psi(self.md_traj)
        return (phi_angles, psi_angles)
