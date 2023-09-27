prefix = "stability_test/tests/data/stability_testing"


def test_generate_visualization():
    from stability_test.vis import MonitoringPlotter

    prefix_path = f"{prefix}/ZINC00061095/"
    s = MonitoringPlotter(
        f"{prefix_path}/vacuum_ZINC00061095_ani2x_nnpops_300.dcd",
        f"{prefix_path}/vacuum_ZINC00061095_ani2x_nnpops_300.pdb",
        f"{prefix_path}/vacuum_ZINC00061095_ani2x_nnpops_300.csv",
    )
    s.set_nglview()
    s.generate_summary(rdf=False)


def test_visualize_DOF_scan():
    from stability_test.vis import MonitoringPlotter

    prefix_path = f"{prefix}/ethanol/"
    s = MonitoringPlotter(
        f"{prefix_path}/vacuum_ethanol_ani2x_nnpops.dcd",
        f"{prefix_path}/vacuum_ethanol_ani2x_nnpops.pdb",
        f"{prefix_path}/vacuum_ethanol_ani2x_nnpops.csv",
    )
    s.set_nglview()
    s.generate_summary(bonded_scan=True)


def test_waterbox():
    from stability_test.vis import MonitoringPlotter

    system_name = "waterbox"
    prefix_path = f"{prefix}/{system_name}/"
    ensemble = "NVT"
    nnp = "ani2x"
    implementation = "nnpops"
    s = MonitoringPlotter(
        f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.dcd",
        f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.pdb",
        f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.csv",
    )

    s.set_nglview(wrap=True, periodic=True)
    s.nglview.add_representation("licorice", selection="water")
    s.generate_summary(water_bond_length=True, rdf=True)
