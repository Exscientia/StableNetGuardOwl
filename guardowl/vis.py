from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import mdtraj as md
import nglview as nv
import numpy as np
import pandas as pd

from guardowl.analysis import PropertyCalculator
from loguru import logger as log
import matplotlib as mpl

mpl.rcParams["figure.constrained_layout.use"] = True


class MonitoringPlotter:
    """
    Generates an interactive plot that visualizes the trajectory and given observable side by side
    """

    def __init__(self, traj_file: str, top_file: str, data_file: str) -> None:
        self.data_file = data_file
        self.canvas = widgets.Output()
        self.md_traj_instance = md.load(traj_file, top=top_file)
        self.x_label_names = ['#"Step"', "Time (ps)", "bond distance [A]"]
        self.data = self._set_data(data_file)
        self.property_calculator = PropertyCalculator(self.md_traj_instance)

        nr_of_waters = len(self.md_traj_instance.top.select("water"))
        nr_of_atoms = self.md_traj_instance.top.n_atoms

        if len(self.md_traj_instance.top.select("water")) > 0:
            self.water_present = True
        else:
            self.water_present = False

        if nr_of_waters == nr_of_atoms:
            log.debug("water only system")
            self.water_only_system = True
        else:
            self.water_only_system = False

        if len(self.md_traj_instance.top.select("resname ALA")) > 0:
            self.dipeptide = True
        else:
            self.dipeptide = False

    def set_nglview(
        self, superpose: bool = False, periodic: bool = False, wrap: bool = False
    ) -> None:
        """generates the nglview trajectory visualizing instance

        Args:
            superpose (bool, optional): superpose the trajectory. Defaults to False.
            periodic (bool, optional): show periodic boundary conditions. Defaults to False.
            wrap (bool, optional): wrap the trajectory. Defaults to False.
        """
        traj = self.md_traj_instance
        if superpose:
            traj.superpose(traj)
        if wrap:
            traj.make_molecules_whole()
        nglview = nv.show_mdtraj(traj)
        if periodic == True:
            nglview.add_unitcell()  # pylint: disable=maybe-no-member
        nglview.center()
        nglview.camera = "orthographic"

        self.nglview = nglview

    def _set_data(self, data_file: str) -> pd.DataFrame:
        """reads in the data

        Args:
            data_file (str): csv file
        """
        with open(data_file) as f:
            data = pd.read_csv(f)
        return data

    def _generate_report_data(self, bonded_scan: bool) -> Tuple[list, list]:
        # read for each observable the label and data
        labels = []
        observable_data = []
        for obs in self.data.keys():
            if obs in self.x_label_names:
                continue
            labels.append(obs)
            if "Total Energy" in obs:
                observable_data.append(np.log(self.data[obs] * -1) * -1)
            else:
                observable_data.append(self.data[obs])

        if self.water_present is True:
            labels.append("water-rdf")
            observable_data.append(self.property_calculator.calculate_water_rdf())
        if self.water_present is True:
            labels.append("water-bond-length")
            observable_data.append(self.property_calculator.monitor_water_bond_length())
        if self.water_present is True:
            labels.append("water-angle")
            observable_data.append(self.property_calculator.monitor_water_angle())
        if self.dipeptide is True:
            labels.append("phi/psi")
            observable_data.append(self.property_calculator.monitor_phi_psi())
        if bonded_scan is False and self.water_only_system is False:
            labels.append("bond deviation")
            observable_data.append(
                self.property_calculator.monitor_bond_length_except_water()
            )

        return labels, observable_data

    def generate_summary(
        self,
        bonded_scan: bool = False,
    ) -> widgets.HBox:
        """Generates the interactive plot

        Returns:
            _type_: _description_
        """

        # generate x axis labels
        nr_of_frames = -1
        try:
            nr_of_frames = len(self.data['#"Step"'])
        except KeyError as e:
            log.debug(e)
            try:
                nr_of_frames = len(self.data["bond distance [A]"])
            except KeyError as e:
                log.debug(e)

        assert nr_of_frames > 0, f"No frames found in data file: {self.data_file}"
        frames = [idx for idx in range(nr_of_frames)]

        labels, observable_data = self._generate_report_data(bonded_scan=bonded_scan)
        label_to_data_map = {labels[i]: observable_data[i] for i in range(len(labels))}
        # generate the subplots
        with self.canvas:
            if bonded_scan:
                fig, axs = plt.subplots(
                    1,
                    1,
                    constrained_layout=True,
                    figsize=(10, 5),
                )
            else:
                fig, axs = plt.subplots(
                    max(int((len(labels) / 3) + 1), 2),
                    3,
                    constrained_layout=True,
                    figsize=(10, 5),
                )

        # move the toolbar to the bottom
        fig.canvas.toolbar_position = "bottom"
        # fig.grid(True)

        # fill the data in the subplots
        lines = []
        column, row = 0, 0
        for l, d in zip(labels, observable_data):
            if l == "water-rdf":
                axs[row][column].plot(*d, "o", alpha=0.5, markersize=2)
                axs[row][column].plot(*d, lw=2)
                axs[row][column].set_xlabel("$r(nm)$")
                axs[row][column].set_ylabel("$g(r)$")
                axs[row][column].set_title("water-rdf")
            elif l == "water-bond-length":
                axs[row][column].hist(d.flatten() * 10)
                axs[row][column].set_title("water O-H bond length")
                axs[row][column].set_xlabel("d [A]")
            elif l == "water-angle":
                axs[row][column].hist(d.flatten())
                axs[row][column].set_title("water H-O-H angle")
                axs[row][column].set_xlabel("angle [degrees]")
            elif l == "bond deviation":
                axs[row][column].hist(d.flatten() * 10)
                axs[row][column].set_title("bond deviation to initial length")
                axs[row][column].set_xlabel("d [A]")
            elif l == "phi/psi":
                axs[row][column].scatter(
                    d[0], d[1], marker="x", c=self.md_traj_instance.time
                )
                axs[row][column].set_title("dihedral map")
                axs[row][column].set_xlabel(r"$\Phi$ Angle [radians]")
                axs[row][column].set_ylabel(r"$\Psi$ Angle [radians]")
                axs[row][column].set_xlim(-np.pi, np.pi)
                axs[row][column].set_ylim(-np.pi, np.pi)
            else:
                if bonded_scan:
                    lines.append(axs.axvline(x=0, color="r", lw=2))
                    axs.plot(frames, d, label=l)
                    axs.set_xticks(
                        np.arange(0, len(frames), 10),
                        [np.round(f, 2) for f in self.data["bond distance [A]"][::10]],
                    )
                    axs.set_xlabel("bond distance [A]")
                else:
                    lines.append(axs[row][column].axvline(x=0, color="r", lw=2))
                    axs[row][column].plot(frames, d, label=l)
                    axs[row][column].set_title(l)
            column += 1
            if column > 2:
                column = 0
                row += 1

        # print numerical values
        try:
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            textstr = "\n".join(
                (
                    r"Heat capacity$=%.12f$ mol/(K J)"
                    % (
                        self.property_calculator.calculate_heat_capacity(
                            label_to_data_map["Total Energy (kJ/mole)"],
                            label_to_data_map["Box Volume (nm^3)"],
                        )._value,
                    ),
                    r"$\kappa_{T}=%.12f$ / bar"
                    % (
                        self.property_calculator.calculate_isothermal_compressability_kappa_T(),
                    ),
                )
            )
            axs[row][column].set_axis_off()
            axs[row][column].text(
                0.05,
                0.95,
                textstr,
                transform=axs[row][column].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )
        except (KeyError, AttributeError) as e:
            log.debug(e)

        try:
            if column == 0:
                axs[row][1].set_axis_off()
                axs[row][2].set_axis_off()
            if column == 1:
                axs[row][2].set_axis_off()
        except TypeError as e:
            log.debug(e)

        fig.tight_layout()
        plt.gca().set_title("title")
        plt.show()

        # callback functions
        def _update(change: str):  # type: ignore
            """redraw line (update plot)"""
            for l in lines:
                l.set_xdata(change.new)  # type: ignore
            fig.canvas.draw()

        # connect callbacks and traits
        self.nglview.observe(_update, "frame")

        return widgets.HBox([self.nglview, self.canvas])
