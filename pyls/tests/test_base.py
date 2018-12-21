# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls


def test_BasePLS(pls_inputs):
    basepls = pyls.base.BasePLS(**pls_inputs)
    for key in pls_inputs.keys():
        assert hasattr(basepls.inputs, key)
        assert np.all(getattr(basepls.inputs, key) == pls_inputs[key])

    with pytest.raises(NotImplementedError):
        basepls.gen_covcorr(pls_inputs['X'], pls_inputs['Y'])
    with pytest.raises(NotImplementedError):
        basepls.gen_distrib(pls_inputs['X'], pls_inputs['Y'])
    with pytest.raises(ValueError):
        pls_inputs['groups'] = [100, 100]
        basepls = pyls.base.BasePLS(**pls_inputs)
