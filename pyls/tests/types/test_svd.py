# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls

Xf = 1000
Yf = 100
subj = 100
rs = np.random.RandomState(1234)


class PLSSVDTest():
    defaults = pyls.structures.PLSInputs(X=rs.rand(subj, Xf),
                                         Y=rs.rand(subj, Yf),
                                         groups=None, n_cond=1,
                                         mean_centering=0, rotate=True,
                                         n_perm=20, n_boot=10, n_split=None,
                                         ci=95, seed=rs, verbose=False)
    funcs = dict(meancentered=pyls.meancentered_pls,
                 behavioral=pyls.behavioral_pls)

    def __init__(self, plstype, **kwargs):
        self.inputs = pyls.structures.PLSInputs(**{key: kwargs.get(key, val)
                                                   for (key, val) in
                                                   self.defaults.items()})
        self.output = self.funcs.get(plstype)(**self.inputs)
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
            num_lv = min([f for f in [Xf, behavior] if f != 1])
        else:
            behavior = num_lv = dummy

        attrs = [
            ('x_weights', (Xf, num_lv)),
            ('y_weights', (behavior, num_lv)),
            ('singvals', (num_lv,)),
            ('varexp', (num_lv,)),
            ('x_scores', (subj, num_lv)),
            ('y_scores', (subj, num_lv)),
        ]

        return attrs

    def confirm_outputs(self):
        """ Confirms generated outputs are of expected shape / size """
        for (attr, shape) in self.make_outputs():
            assert attr in self.output
            assert self.output[attr].shape == shape


@pytest.mark.parametrize(('n_split', 'rotate'), [
    (None, True), (None, False), (5, True), (5, False)
])
def test_behavioral_onegroup_onecondition(n_split, rotate):
    PLSSVDTest('behavioral', groups=None, n_cond=1, n_split=n_split,
               rotate=rotate)


@pytest.mark.parametrize(('n_split', 'rotate'), [
    (None, True), (None, False), (5, True), (5, False)
])
def test_behavioral_multigroup_onecondition(n_split, rotate):
    PLSSVDTest('behavioral', groups=[33, 34, 33], n_cond=1, n_split=n_split,
               rotate=rotate)


@pytest.mark.parametrize(('n_split', 'rotate'), [
    (None, True), (None, False), (5, True), (5, False)
])
def test_behavioral_onegroup_multicondition(n_split, rotate):
    PLSSVDTest('behavioral', groups=subj // 4, n_cond=4, n_split=n_split,
               rotate=rotate)


@pytest.mark.parametrize(('n_split', 'rotate'), [
    (None, True), (None, False), (5, True), (5, False)
])
def test_behavioral_multigroup_multicondition(n_split, rotate):
    PLSSVDTest('behavioral', groups=[25, 25], n_cond=2, n_split=n_split,
               rotate=rotate)


@pytest.mark.parametrize(('mean_centering', 'n_split', 'rotate'), [
    (1, None, True), (1, None, False), (1, 5, True), (1, 5, False),
    (2, None, True), (2, None, False), (2, 5, True), (2, 5, False)
])
def test_meancentered_multigroup_onecondition(mean_centering, n_split, rotate):
    PLSSVDTest('meancentered', groups=[33, 34, 33], n_cond=1, n_split=n_split,
               mean_centering=mean_centering, rotate=rotate)


@pytest.mark.parametrize(('mean_centering', 'n_split', 'rotate'), [
    (0, None, True), (0, None, False), (0, 5, True), (0, 5, False),
    (2, None, True), (2, None, False), (2, 5, True), (2, 5, False)
])
def test_meancentered_onegroup_multicondition(mean_centering, n_split, rotate):
    PLSSVDTest('meancentered', groups=subj // 2, n_cond=2, n_split=n_split,
               mean_centering=mean_centering, rotate=rotate)


@pytest.mark.parametrize(('mean_centering', 'n_split', 'rotate'), [
    (0, None, True), (0, None, False), (0, 5, True), (0, 5, False),
    (1, None, True), (1, None, False), (1, 5, True), (1, 5, False),
    (2, None, True), (2, None, False), (2, 5, True), (2, 5, False)
])
def test_meancentered_multigroup_multicondition(mean_centering, n_split,
                                                rotate):
    PLSSVDTest('meancentered', groups=[25, 25], n_cond=2, n_split=n_split,
               mean_centering=mean_centering, rotate=rotate)


def test_warnings():
    with pytest.warns(UserWarning):
        PLSSVDTest('meancentered', groups=[50, 50], mean_centering=0)
    with pytest.warns(UserWarning):
        PLSSVDTest('meancentered', n_cond=2, mean_centering=1)


def test_errors():
    with pytest.raises(ValueError):
        PLSSVDTest('meancentered', groups=[50, 50], mean_centering=3)
    with pytest.raises(ValueError):
        PLSSVDTest('meancentered', groups=[subj])
    with pytest.raises(ValueError):
        PLSSVDTest('meancentered', n_cond=3)
    with pytest.raises(ValueError):
        PLSSVDTest('behavioral', Y=rs.rand(subj - 1, Yf))
