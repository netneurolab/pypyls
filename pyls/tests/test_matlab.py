"""Tests for MATLAB equivalence in pyls."""

import pytest
import os
from pyls.tests import assert_matlab_equivalence

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

data_dir = files("pyls") / "tests" / "data"


@pytest.mark.parametrize(
    "mat_file",
    [
        "bpls_multigroup_multicond_nosplit.mat",
        "bpls_multigroup_multicond_split.mat",
        "bpls_multigroup_onecond_nosplit.mat",
        "bpls_multigroup_onecond_split.mat",
        "bpls_onegroup_multicond_nosplit.mat",
        "bpls_onegroup_multicond_split.mat",
        "bpls_onegroup_onecond_nosplit.mat",
        "bpls_onegroup_onecond_split.mat",
        "mpls_multigroup_multicond_nosplit.mat",
        "mpls_multigroup_multicond_split.mat",
        "mpls_multigroup_onecond_nosplit.mat",
        "mpls_multigroup_onecond_split.mat",
        "mpls_onegroup_multicond_nosplit.mat",
        "mpls_onegroup_multicond_split.mat",
    ],
)
def test_matlab_equivalence(mat_file):
    """Test MATLAB equivalence for each .mat file."""
    if not os.path.exists(data_dir / mat_file):
        pytest.skip(f"MATLAB file {mat_file} not found")

    print(f"Testing {mat_file}")
    assert_matlab_equivalence(
        data_dir / mat_file, n_proc="max", n_perm=2500, n_split=100
    )
