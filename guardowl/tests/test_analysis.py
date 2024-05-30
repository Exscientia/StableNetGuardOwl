import mdtraj as md

prefix = "guardowl/tests/data/stability_testing"


def generate_water_mdtraj_instance() -> md.Trajectory:
    system_name = "waterbox"
    prefix_path = f"{prefix}/{system_name}/"
    ensemble = "npt"
    nnp = "ani2x"

    traj_file = f"{prefix_path}/{system_name}_15A_{nnp}_{ensemble}.dcd"
    top_file = f"{prefix_path}/{system_name}_15A_{nnp}_{ensemble}.pdb"
    return md.load(traj_file, top=top_file)


def get_water_csv_file():
    system_name = "waterbox"
    prefix_path = f"{prefix}/{system_name}/"
    ensemble = "npt"
    nnp = "ani2x"

    csv_file = f"{prefix_path}/{system_name}_15A_{nnp}_{ensemble}.csv"
    return csv_file


def test_rdf():
    from guardowl.analysis import PropertyCalculator

    md_traj_instance = generate_water_mdtraj_instance()
    property_calculator = PropertyCalculator(md_traj_instance)

    assert len(md_traj_instance.top.select("water")) > 0

    rdf = property_calculator.calculate_water_rdf()


def test_calculate_properties():
    from guardowl.analysis import PropertyCalculator
    import numpy as np

    md_traj_instance = generate_water_mdtraj_instance()
    csv_file = get_water_csv_file()
    # read and extract columns from csv file
    csv_data = np.genfromtxt(csv_file, delimiter=",", names=True)
    total_energy = csv_data["Total_Energy_kJmole"]
    volumn = csv_data["Box_Volume_nm3"]

    property_calculator = PropertyCalculator(md_traj_instance)
    property_calculator.calculate_heat_capacity(total_energy, volumn)
    property_calculator.calculate_isothermal_compressability_kappa_T()
