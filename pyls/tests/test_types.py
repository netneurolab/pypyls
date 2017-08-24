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

attrs = ['U','d','V',
         'd_pvals','d_kaiser','d_varexp',
         'U_bci','V_bci',
         'U_bsr','V_bsr',
         'U_sig','V_sig']


def test_behavioral_pls():
    o1 = pyls.types.behavioral_pls(braindata, behavmat, comp, n_perm, n_boot)
    o2 = pyls.types.behavioral_pls(behavmat, braindata, comp, n_perm, n_boot)
    for f in attrs: assert hasattr(o1,f)

    with pytest.raises(ValueError):
        pyls.types.behavioral_pls(behavmat[:,0], braindata, comp)
    with pytest.raises(ValueError):
        pyls.types.behavioral_pls(behavmat[:,0], braindata[:,0], comp)

def test_group_behavioral_pls():
    pyls.types.behavioral_pls(groupbraindata, groupbehavmat,
                              groups, n_perm, n_boot)

    onecol = np.stack([np.ones([comp,1]),np.ones([comp,1])*2], axis=2)

    with pytest.raises(ValueError):
        pyls.types.behavioral_pls(groupbraindata, onecol, 5,
                                  n_perm=n_perm, n_boot=n_boot)

    pyls.types.behavioral_pls(groupbraindata, onecol,
                              n_perm=n_perm, n_boot=n_boot)
