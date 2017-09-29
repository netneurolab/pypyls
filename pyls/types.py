#!/usr/bin/env python

import numpy as np
from pyls import compute


class behavioral_pls():
    """
    Runs PLS on `brain` and `behav` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    cross-covariance matrix of `brain` and `behav`.

    Parameters
    ----------
    brain : array_like
        Array of data in shape (N x k [x group])
    behav : array_like
        Array of data in shape (N x j [x group])
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during permutation testing. Default: 100
    p : float, optional
        Signifiance criterion for bootstrapping, within (0, 1). Default: 0.05
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Attributes
    -------
    U : array
        left singular vectors (k[*group] x n_comp)
    d : array
        singular values (diagonal, n_comp x n_comp)
    V : array
        right singular vectors (j x n_comp)
    d_pvals : array
        p-values of latent variables as determined by permutation (n_comp)
    d_kaiser : array
        relevance of latent variables as determined by Kaiser criterion
        (n_comp)
    d_varexp : array
        percent variance explained by each latent variable (n_comp)
    U_bci : array
        bootstrapped CI for left singular vectors (k x n_comp x 2)
    V_bci : array
        bootstrapped CI for right singular vectors (j x n_comp x 2)
    U_bsr : array
        bootstrap ratios for left singular vectors (k x n_comp)
    V_bsr : array
        bootstrap ratios for right singular vectors (j x n_comp)
    U_sig : array
        significance (by zero-crossing) of left singular vectors (k x n_comp)
    V_sig : array
        significance (by zero-crossing) of right singular vectors (j x n_comp)
    """

    def __init__(self, brain, behav,
                 n_perm=5000, n_boot=1000,
                 n_split=None,
                 p=0.05,
                 seed=None):
        self.brain, self.behav = brain, behav
        self._n_perm, self._n_boot, self._n_split = n_perm, n_boot, n_split
        self._p = p

        if seed is not None: np.random.seed(seed)

        self.run_svd()
        self.run_perms()
        self.run_boots()
        self.get_sig()

    def run_svd(self):
        self.U, self.d, self.V = compute.svd(self.brain,
                                             self.behav)

        self._n_comp = len(self.d)

    def run_perms(self):
        if len(self.U) < len(self.V): orig = self.U
        else: orig = self.V

        perms = compute.serial_permute(self.brain, self.behav,
                                       self._n_comp, orig,
                                       n_perm=self._n_perm,
                                       n_split=self._n_split)
        self.d_pvals = compute.perm_sig(perms, self.d)

    def run_boots(self):
        U_boot, V_boot = compute.bootstrap(self.brain, self.behav,
                                           self._n_comp,
                                           self.U, self.V,
                                           n_boot=self._n_boot)
        self.U_bci, self.V_bci = compute.boot_ci(U_boot, V_boot, p=self._p)
        self.U_bsr, self.V_bsr = compute.boot_rel(self.U, self.V,
                                                  U_boot, V_boot)

    def get_sig(self):
        self.U_sig = compute.boot_sig(self.U_bci)
        self.V_sig = compute.boot_sig(self.V_bci)

        self.d_kaiser = compute.kaiser_criterion(self.d)
        self.d_varexp = compute.crossblock_cov(self.d)
