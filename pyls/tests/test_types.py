#!/usr/bin/env python

import pytest
import numpy as np
import pyls

brain    = 1000
behavior = 100
comp     = 20
n_perm   = 50
n_boot   = 10
groups   = 2

behavmat  = np.random.rand(comp,behavior)
braindata = np.random.rand(comp,brain)

groupbehavmat  = np.random.rand(comp, behavior, groups)
groupbraindata = np.random.rand(comp, brain, groups)


def test_behavioral_pls():
    pyls.types.behavioral_pls(braindata, behavmat, comp, n_perm, n_boot)
    pyls.types.behavioral_pls(behavmat, braindata, comp, n_perm, n_boot)
    with pytest.raises(ValueError):
        pyls.types.behavioral_pls(behavmat[:,0], braindata, comp)
    with pytest.raises(ValueError):
        pyls.types.behavioral_pls(behavmat[:,0], braindata[:,0], comp)


def test_group_behavioral_pls():
    pyls.types.behavioral_pls(groupbehavmat, groupbraindata,
                              groups, n_perm, n_boot)
