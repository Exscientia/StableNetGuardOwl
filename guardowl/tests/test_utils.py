def test_input_expension_for_pure_liquids():
    molecule_name = ["ethane", "butane"]
    nr_of_molecules = [100, 200, 300]
    molecule_name_ = molecule_name * len(nr_of_molecules)
    nr_of_molecule_ = [
        element for element in nr_of_molecules for _ in range(len(molecule_name))
    ]
    counter = 0
    for i, j in zip(molecule_name_, nr_of_molecule_):
        print(i, j)
        counter += 1
    assert counter == 6


def test_tar_gz_drugbank_extraction():
    from guardowl.utils import extract_drugbank_tar_gz
    extract_drugbank_tar_gz()
