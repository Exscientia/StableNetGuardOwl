from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import mdtraj as md
import nglview as nv
import numpy as np
import pandas as pd

from stability_test.analysis import PropertyCalculator
import loguru as logger


class MonitoringPlotter:
    """
    Generates an interactive plot that visualizes the trajectory and given observable side by side
    """

    def __init__(self, traj_file: str, top_file: str, data_file: str) -> None:
        self.canvas = widgets.Output()
        self.md_traj_instance = md.load(traj_file, top=top_file)
        self.x_label_names = ['#"Step"', "Time (ps)", "bond distance [A]"]
        self.data = self._set_data(data_file)
        self.property_calculator = PropertyCalculator(self.md_traj_instance)

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

    def _generate_report_data(
        self, rdf: bool, water_bond_length: bool, water_angle: bool
    ) -> Tuple[list, list]:
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

        if rdf is True:
            labels.append("water-rdf")
            observable_data.append(self.property_calculator.calculate_water_rdf())
        if water_bond_length is True:
            labels.append("water-bond-length")
            observable_data.append(self.property_calculator.monitor_water_bond_length())
        if water_angle is True:
            labels.append("water-angle")
            observable_data.append(self.property_calculator.monitor_water_angle())

        return labels, observable_data

    def generate_summary(
        self,
        bonded_scan: bool = False,
        rdf: bool = False,
        water_bond_length: bool = False,
        water_angle: bool = False,
    ) -> widgets.HBox:
        """Generates the interactive plot

        Returns:
            _type_: _description_
        """

        if bonded_scan is True:
            assert (rdf or water_angle or water_bond_length) is False

        # generate x axis labels
        if '#"Step"' in self.data.keys():
            frames = [idx for idx, _ in enumerate(self.data['#"Step"'])]
        elif "bond distance [A]" in self.data.keys():
            # frames = self.data["bond distance [A]"]
            frames = [idx for idx, _ in enumerate(self.data["bond distance [A]"])]

        labels, observable_data = self._generate_report_data(
            rdf, water_bond_length, water_angle
        )
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
                axs[row][column].hist(d.flatten())
                axs[row][column].set_title("water O-H bond length")
            elif l == "water-angle":
                axs[row][column].hist(d.flatten())
                axs[row][column].set_title("water H-O-H angle")
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

        fig.tight_layout()
        plt.gca().set_title("title")

        # callback functions
        def _update(change: str):  # type: ignore
            """redraw line (update plot)"""
            for l in lines:
                l.set_xdata(change.new)  # type: ignore
            fig.canvas.draw()

        # connect callbacks and traits
        self.nglview.observe(_update, "frame")

        return widgets.HBox([self.nglview, self.canvas])
