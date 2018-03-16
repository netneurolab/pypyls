# -*- coding: utf-8 -*-

import numpy as np
import pyls

Xf = 1000
Yf = 100
subj = 100
rs = np.random.RandomState(1234)


class PLSBaseTest():
    defaults = pyls.base.PLSInputs(X=rs.rand(subj, Xf),
                                   Y=rs.rand(subj, Yf),
                                   groups=[subj],
                                   n_cond=1,
                                   mean_centering=0,
                                   rotate=True,
                                   n_perm=20, n_boot=10, n_split=None,
                                   ci=95, seed=rs)
    funcs = dict(meancentered=pyls.MeanCenteredPLS,
                 behavioral=pyls.BehavioralPLS)

    def __init__(self, plstype, **kwargs):
        # confirm input
        if plstype not in self.funcs.keys():
            raise ValueError('Argument `plstype` must be in {}'.format(
                list(self.funcs.keys())))

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
        attrs : list
            Expected attributes and shapes
        """

        if self.type == 'behavioral':
            behavior = Yf * len(self.inputs.groups) * self.inputs.n_cond
            num_lv = min(subj, Xf, Yf)
        else:
            behavior = len(self.inputs.groups) * self.inputs.n_cond
            num_lv = len(self.inputs.groups) * self.inputs.n_cond

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
        """
        Used to confirm ``output`` has expected ``attributes```

        Parameters
        ----------
        output : PLS output
        attributes : list-of-tuple
            From output of ``make_outputs()``
        """
        attributes = self.make_outputs()
        for (attr, shape) in attributes:
            assert hasattr(self.output, attr)
            assert getattr(self.output, attr).shape == shape


def test_BehavioralPLS_onegroup_onecond():
    for n_split in [None, 5]:
        PLSBaseTest('behavioral',
                    groups=[subj],
                    n_cond=1,
                    n_split=n_split)


def test_BehavioralPLS_multigroup_onecond():
    for n_split in [None, 5]:
        PLSBaseTest('behavioral',
                    groups=[33, 34, 33],
                    n_cond=1,
                    n_split=n_split)


def test_BehavioralPLS_onegroup_multicond():
    for n_split in [None, 5]:
        PLSBaseTest('behavioral',
                    groups=[subj],
                    n_cond=4,
                    n_split=n_split)


def test_BehavioralPLS_multigroup_multicond():
    for n_split in [None, 5]:
        PLSBaseTest('behavioral',
                    groups=[25, 25],
                    n_cond=2,
                    n_split=n_split)


def test_MeanCenteredPLS_multigroup_onecond():
    for n_split in [None, 5]:
        PLSBaseTest('meancentered',
                    groups=[33, 34, 33],
                    n_cond=1,
                    n_split=n_split)


def test_MeanCenteredPLS_onegroup_multicond():
    for n_split in [None, 5]:
        PLSBaseTest('meancentered',
                    groups=[subj],
                    n_cond=2,
                    n_split=n_split)


def test_MeanCenteredPLS_multigroup_multicond():
    for n_split in [None, 5]:
        PLSBaseTest('meancentered',
                    groups=[25, 25],
                    n_cond=2,
                    n_split=n_split)
