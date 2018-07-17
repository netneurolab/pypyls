# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
import pyls

Xf = 1000
Yf = 100
subj = 100
rs = np.random.RandomState(1234)


class PLSBaseTest():
    defaults = pyls.base.PLSInputs(X=rs.rand(subj, Xf),
                                   Y=rs.rand(subj, Yf),
                                   groups=None,
                                   n_cond=1,
                                   mean_centering=0,
                                   rotate=True,
                                   n_perm=20, n_boot=10, n_split=None,
                                   ci=95, seed=rs)
    funcs = dict(meancentered=pyls.MeanCenteredPLS,
                 behavioral=pyls.BehavioralPLS)

    def __init__(self, plstype, **kwargs):
        self.inputs = pyls.base.PLSInputs(**{key: kwargs.get(key, val) for
                                             (key, val) in
                                             self.defaults.items()})
        self.output = self.funcs.get(plstype)(**self.inputs).results
        self.type = plstype
        self.confirm_outputs()

    def make_outputs(self):
        """
        Used to make list of expected attributes and shapes for PLS outputs

        Returns
        -------
        attrs : list-of-tuples
            Each entry in the list is a tuple with the attribute name and
            expected shape
        """

        dummy = len(self.output.inputs.groups) * self.output.inputs.n_cond
        if self.type == 'behavioral':
            behavior = Yf * dummy
            num_lv = min([f for f in [subj, Xf, Yf, dummy] if f != 1])
        else:
            behavior = num_lv = dummy

        attrs = [
            ('u', (Xf, num_lv)),
            ('s', (num_lv, num_lv)),
            ('v', (behavior, num_lv)),
            ('usc', (subj, num_lv)),
            ('vsc', (subj, num_lv)),
            ('s_varexp', (num_lv,))
        ]

        return attrs

    def confirm_outputs(self):
        """ Confirms generated outputs are of expected shape / size """
        for (attr, shape) in self.make_outputs():
            assert hasattr(self.output, attr)
            assert getattr(self.output, attr).shape == shape


def test_BehavioralPLS_onegroup_onecondition():
    kwargs = dict(groups=None, n_cond=1)
    for (ns, rt) in itertools.product([None, 5], [True, False]):
        PLSBaseTest('behavioral', n_split=ns, rotate=rt, **kwargs)


def test_BehavioralPLS_multigroup_onecondition():
    kwargs = dict(groups=[33, 34, 33], n_cond=1)
    for (ns, rt) in itertools.product([None, 5], [True, False]):
        PLSBaseTest('behavioral', n_split=ns, rotate=rt, **kwargs)


def test_BehavioralPLS_onegroup_multicondition():
    kwargs = dict(groups=subj // 4, n_cond=4)
    for (ns, rt) in itertools.product([None, 5], [True, False]):
        PLSBaseTest('behavioral', n_split=ns, rotate=rt, **kwargs)


def test_BehavioralPLS_multigroup_multicondition():
    kwargs = dict(groups=[25, 25], n_cond=2)
    for (ns, rt) in itertools.product([None, 5], [True, False]):
        PLSBaseTest('behavioral', n_split=ns, rotate=rt, **kwargs)


def test_MeanCenteredPLS_multigroup_onecondition():
    kwargs = dict(groups=[33, 34, 33], n_cond=1)
    for (mc, ns, rt) in itertools.product([1, 2], [None, 5], [True, False]):
        PLSBaseTest('meancentered', n_split=ns, mean_centering=mc, rotate=rt,
                    **kwargs)
    with pytest.warns(UserWarning):
        PLSBaseTest('meancentered', groups=[50, 50], mean_centering=0)


def test_MeanCenteredPLS_onegroup_multicondition():
    kwargs = dict(groups=[subj // 2], n_cond=2)
    for (mc, ns, rt) in itertools.product([0, 2], [None, 5], [True, False]):
        PLSBaseTest('meancentered', n_split=ns, mean_centering=mc, rotate=rt,
                    **kwargs)
    with pytest.warns(UserWarning):
        PLSBaseTest('meancentered', mean_centering=1, **kwargs)


def test_MeanCenteredPLS_multigroup_multicondition():
    kwargs = dict(groups=[25, 25], n_cond=2)
    for (mc, ns, rt) in itertools.product([0, 1, 2], [None, 5], [True, False]):
        PLSBaseTest('meancentered', n_split=ns, mean_centering=mc, rotate=rt,
                    **kwargs)


def test_errors():
    with pytest.raises(ValueError):
        PLSBaseTest('meancentered', groups=[50, 50], mean_centering=3)
    with pytest.raises(ValueError):
        PLSBaseTest('meancentered', groups=[subj])
