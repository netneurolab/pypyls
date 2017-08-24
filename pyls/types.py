#!/usr/bin/env python

import numpy as np
from pyls import compute


class behavioral_pls():
    """
    Runs behavioral PLS on `brain` and `behav` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    cross-covariance matrix of `brain` and `behav`.

    Parameters
    ----------
    brain, behav : array (N x k [x group]), array (N x j [x group])
        provided data
    n_comp : int
        number of components to return from SVD (default: rank of Y.T @ X)
    n_perm : int
        number of permutations to generate (default: 5000)
    n_boot : int
        number of bootstraps to generate (default: 1000)
    p : float (0,1)
        signifiance criterion for bootstrapping (default: 0.01)
    seed : int
        whether to set random seed for reproducibility (default: None)

    Attributes
    -------
    U : array (k[*group] x n_comp)
        left singular vectors
    d : array (diagonal, n_comp x n_comp)
        singular values
    V : array (j x n_comp)
        right singular vectors
    d_pvals : array (by permutation; n_comp)
        p-values of latent variables
    d_kaiser : array (by Kaiser criterion; n_comp)
        relevance of latent variables
    d_varexp : array (n_comp)
        percent variance explained by each latent variable
    U_bci : array (k x n_comp x 2)
        bootstrapped CI for left singular vectors
    V_bci : array (j x n_comp x 2)
        bootstrapped CI for right singular vectors
    U_bsr : array (k x n_comp)
        bootstrap ratios for left singular vectors
    V_bsr : array (j x n_comp)
        bootstrap ratios for right singular vectors
    U_sig : array (k x n_comp)
        significance (by zero-crossing) of left singular vectors
    V_sig : array (j x n_comp)
        significance (by zero-crossing) of right singular vectors
    """

    def __init__(self, brain, behav, n_comp=None,
                 n_perm=5000, n_boot=1000, p=0.01,
                 seed=None):
        self.brain, self.behav = brain, behav
        self._n_comp = n_comp
        self._n_perm, self._n_boot = n_perm, n_boot
        self._p = p

        if seed is not None: np.random.seed(seed)

        self._run_svd()
        self._run_perms()
        self._run_boots()
        self._get_sig()

    def _run_svd(self):
        self.U, self.d, self.V = compute.svd(self.brain,
                                             self.behav,
                                             self._n_comp)

        if self._n_comp is None: self._n_comp = len(self.d)

    def _run_perms(self):
        if len(self.U) < len(self.V): orig = self.U
        else: orig = self.V

        perms = compute.serial_permute(self.brain, self.behav,
                                       self._n_comp, orig,
                                       n_perm=self._n_perm)
        self.d_pvals = compute.perm_sig(perms, self.d)

    def _run_boots(self):
        U_boot, V_boot = compute.bootstrap(self.brain, self.behav,
                                           self._n_comp,
                                           self.U, self.V,
                                           n_boot=self._n_boot)
        self.U_bci, self.V_bci = compute.boot_ci(U_boot, V_boot, p=self._p)
        self.U_bsr, self.V_bsr = compute.boot_rel(self.U, self.V,
                                                  U_boot, V_boot)

    def _get_sig(self):
        self.U_sig = compute.boot_sig(self.U_bci)
        self.V_sig = compute.boot_sig(self.V_bci)

        self.d_kaiser = compute.kaiser_criterion(self.d)
        self.d_varexp = compute.crossblock_cov(self.d)
