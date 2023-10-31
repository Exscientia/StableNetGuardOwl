import mdtraj as md

prefix = "guardowl/tests/data/stability_testing"


def generate_water_mdtraj_instance() -> md.Trajectory:
    system_name = "waterbox"
    prefix_path = f"{prefix}/{system_name}/"
    ensemble = "NVT"
    nnp = "ani2x"
    implementation = "nnpops"

    traj_file = f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.dcd"
    top_file = f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.pdb"
    return md.load(traj_file, top=top_file)


def get_water_csv_file():
    system_name = "waterbox"
    prefix_path = f"{prefix}/{system_name}/"
    ensemble = "NVT"
    nnp = "ani2x"
    implementation = "nnpops"

    csv_file = f"{prefix_path}/{system_name}_15A_{nnp}_{implementation}_{ensemble}.csv"
    return csv_file


def test_rdf():
    from guardowl.analysis import PropertyCalculator

    md_traj_instance = generate_water_mdtraj_instance()
    property_calculator = PropertyCalculator(md_traj_instance)

    assert len(md_traj_instance.top.select("water")) > 0

    rdf = property_calculator.calculate_water_rdf()
    print(rdf)


def test_calculate_properties():
    from guardowl.analysis import PropertyCalculator
    import numpy as np

    md_traj_instance = generate_water_mdtraj_instance()
    csv_file = get_water_csv_file()
    # read and extract columns from csv file
    csv_data = np.genfromtxt(csv_file, delimiter=",", names=True)
    total_energy = csv_data["Total_Energy_kJmole"]
    # volumn = csv_data["Volume_nm3"]
    N = len(total_energy)  # Length of the array
    mean = 15  # mean volumn
    stddev = 1

    volumn = np.random.normal(mean, stddev, N)

    property_calculator = PropertyCalculator(md_traj_instance)
    heat_capacity = property_calculator.calculate_heat_capacity(total_energy, volumn)
    print(heat_capacity)
    