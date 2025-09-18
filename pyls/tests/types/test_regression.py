"""Tests for PLS regression methods."""

import numpy as np
import pytest
import pyls

Xf = 100
Yf = 100
subj = 50
rs = np.random.RandomState(1234)


class PLSRegressionTests:
    """Class to test PLS regression methods with various parameters."""

    defaults = pyls.structures.PLSInputs(
        X=rs.rand(subj, Xf),
        Y=rs.rand(subj, Yf),
        n_perm=1,
        n_boot=1,
        ci=95,
        seed=rs,
        n_proc="max",
        verbose=True,
    )

    def __init__(self, n_components=None, **kwargs):
        params = self.defaults.copy()
        params.update(kwargs)
        self.inputs = pyls.structures.PLSInputs(**params)
        self.inputs["n_components"] = n_components
        self.output = pyls.pls_regression(**self.inputs)
        self.confirm_outputs()

    def make_outputs(self):
        """
        Make list of expected attributes and shapes for PLS outputs.

        Returns
        -------
        attrs : list-of-tuples
            Each entry in the list is a tuple with the attribute name and
            expected shape
        """
        if self.inputs["n_components"] is None:
            num_lv = subj - 1
        else:
            num_lv = self.inputs["n_components"]

        attrs = [
            ("x_weights", (Xf, num_lv)),
            ("x_scores", (subj, num_lv)),
            ("y_scores", (subj, num_lv)),
            ("y_loadings", (Yf, num_lv)),
            ("varexp", (num_lv,)),
        ]

        return attrs

    def confirm_outputs(self):
        """Confirm generated outputs are of expected shape / size."""
        for attr, shape in self.make_outputs():
            assert attr in self.output
            assert self.output[attr].shape == shape


@pytest.mark.parametrize("n_components", [None, 2, 5, 10, 15])
def test_regression_onegroup_onecondition(n_components):
    """Test PLS regression with one group and one condition."""
    PLSRegressionTests(n_components=n_components)


@pytest.mark.xfail()
@pytest.mark.parametrize("aggfunc", ["mean", "median", "sum"])
def test_regression_3dbootstrap(aggfunc):
    """Test PLS regression with 3D bootstrap arrays and aggregation functions."""
    # confirm providing 3D arrays works
    Y = rs.rand(subj, Yf, 100)
    PLSRegressionTests(Y=Y, n_components=2, aggfunc=aggfunc)

    # confirm providing valid bootsamples for 3D array works
    sboot = pyls.base.gen_bootsamp([subj], 1, n_boot=10)
    nboot = pyls.base.gen_bootsamp([100], 1, n_boot=10)
    bootsamples = np.array(list(zip(sboot.T, nboot.T))).T
    PLSRegressionTests(
        Y=Y, n_components=2, aggfunc=aggfunc, bootsamples=bootsamples, n_boot=10
    )


def test_regression_missingdata():
    """Test PLS regression with missing data in X and Y matrices."""
    X = rs.rand(subj, Xf)
    X[10] = np.nan
    PLSRegressionTests(X=X, n_components=2)
    X[20] = np.nan
    PLSRegressionTests(X=X, n_components=2)
    Y = rs.rand(subj, Yf)
    Y[11] = np.nan
    PLSRegressionTests(X=X, Y=Y, n_components=2)


@pytest.mark.xfail()
def test_errors():
    """Test error handling for invalid input parameters."""
    with pytest.raises(ValueError):
        PLSRegressionTests(n_components=1000)
    with pytest.raises(ValueError):
        PLSRegressionTests(Y=rs.rand(subj - 1, Yf))
    with pytest.raises(ValueError):
        PLSRegressionTests(Y=rs.rand(subj, Yf, 10), aggfunc="notafunc")
    with pytest.raises(TypeError):
        PLSRegressionTests(Y=rs.rand(subj, Yf, 10), aggfunc=lambda x: x)
    with pytest.raises(ValueError):
        PLSRegressionTests(Y=rs.rand(subj, Yf, 10), bootsamples=[[10], [10]])
