"""Tests for the examples module."""

import os
import pytest
import pyls.examples

DATASETS = ["mirchi_2018", "whitaker_vertes_2016", "wine", "linnerud"]


def test_available_datasets():
    """Test the available_datasets function for listing and validating dataset names."""
    # make sure we get a list of strings when called with no arguments
    avail = pyls.examples.available_datasets()
    assert isinstance(avail, list)
    assert all([isinstance(f, str) for f in avail])

    # check that we get all expected datasets back
    assert len(set(DATASETS) - set(avail)) == 0

    # check that we can supply dataset names to function to confirm validity
    for f in DATASETS:
        assert f == pyls.examples.available_datasets(f)

    # check that providing non-valid dataset name errors
    for f in ["thisisnotadataset", 10]:
        with pytest.raises(ValueError):
            pyls.examples.available_datasets(f)


@pytest.mark.parametrize(
    ("dataset", "keys"),
    [
        (
            "linnerud",
            ["description", "reference", "urls", "X", "Y", "n_perm", "n_boot"],
        ),
        (
            "mirchi_2018",
            [
                "description",
                "reference",
                "urls",
                "X",
                "Y",
                "n_perm",
                "n_boot",
                "test_size",
                "test_split",
                "parcellation",
            ],
        ),
        (
            "wine",
            ["description", "reference", "urls", "X", "n_perm", "n_boot", "groups"],
        ),
        (
            "whitaker_vertes_2016",
            [
                "description",
                "reference",
                "urls",
                "X",
                "Y",
                "n_perm",
                "n_boot",
                "n_components",
            ],
        ),
    ],
)
def test_query_dataset(dataset, keys):
    """Test querying dataset information for valid keys and data."""
    # check that default return string (description)
    assert isinstance(pyls.examples.query_dataset(dataset), str)
    # check that supplying None returns all available keys
    assert set(pyls.examples.query_dataset(dataset, None)) == set(keys)
    # check that all valid keys return something
    for k in keys:
        assert pyls.examples.query_dataset(dataset, k) is not None
    # check nonsense keys
    for k in ["notakey", 10, 20.5132]:
        with pytest.raises(KeyError):
            pyls.examples.query_dataset(dataset, k)


def test_get_data_dir(tmpdir):
    """Test the _get_data_dir function for directory handling."""
    # check that default (no arguments) returns valid default directory
    data_dir = pyls.examples.datasets._get_data_dir()
    assert isinstance(data_dir, str)
    assert os.path.exists(data_dir)
    assert os.path.basename(data_dir) == "pyls-data"

    # check supplying directory returns same directory
    assert pyls.examples.datasets._get_data_dir(str(tmpdir)) == str(tmpdir)
    assert os.path.exists(str(tmpdir))

    # check that _get_data_dir() pulls from environmental variable correctly
    os.environ["PYLS_DATA"] = str(tmpdir)
    assert pyls.examples.datasets._get_data_dir() == str(tmpdir)


@pytest.mark.parametrize(
    ("dataset", "keys"),
    [
        ("linnerud", ["X", "Y", "n_perm", "n_boot"]),
        ("mirchi_2018", ["X", "Y", "n_perm", "n_boot", "test_size", "test_split"]),
        ("wine", ["X", "groups", "n_perm", "n_boot"]),
        ("whitaker_vertes_2016", ["X", "Y", "n_perm", "n_boot", "n_components"]),
    ],
)
def test_load_dataset(tmpdir, dataset, keys):
    """Test loading datasets and validating PLSInputs structure."""
    ds = pyls.examples.load_dataset(dataset, str(tmpdir))
    assert isinstance(ds, pyls.structures.PLSInputs)
    for k in keys:
        assert hasattr(ds, k) and getattr(ds, k) is not None
    ds, ref = pyls.examples.load_dataset(dataset, str(tmpdir), return_reference=True)
    assert isinstance(ref, str)
