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


def test_rdf():
    from guardowl.analysis import PropertyCalculator

    md_traj_instance = generate_water_mdtraj_instance()
    property_calculator = PropertyCalculator(md_traj_instance)

    assert len(md_traj_instance.top.select("water")) > 0

    rdf = property_calculator.calculate_water_rdf()
    print(rdf)
