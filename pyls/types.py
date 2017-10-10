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
    brain : (N x K [x G]) array_like
        Where `N` is the number of subjects, `K` is the number of observations,
        and `G` is a grouping factor
    behav : (N x J [x G]) array_like
        Where `N` is the number of subjects, `J` is the number of observations,
        and `G` is a grouping factor
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during permutation testing. Default: 100
    p : float (0,1), optional
        Signifiance criterion for bootstrapping, within (0, 1). Default: 0.05
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Attributes
    -------
    U : (K[*G] x N_COMP) ndarray
        Left singular vectors
    d : (N_COMP x N_COM) ndarray
        Diagonal matrix of singular values
    V : (J x N_COMP) ndarray
        Right singular vectors
    d_pvals : (N_COMP,) ndarray
        P-values of latent variables as determined by permutation testing
    d_kaiser : (N_COMP,) ndarray
        Relevance of latent variables as determined by Kaiser criterion
    d_varexp : (N_COMP,) ndarray
        Percent variance explained by each latent variable
    U_bci : (K[*G] x N_COMP x 2) ndarray
        Bootstrapped CI for left singular vectors
    V_bci : (J x N_COMP x 2) ndarray
        Bootstrapped CI for right singular vectors
    U_bsr : (K[*G] x N_COMP) ndarray
        Bootstrap ratios for left singular vectors
    V_bsr : (J x N_COMP) ndarray
        Bootstrap ratios for right singular vectors
    U_sig : (K[*G] x N_COMP) ndarray
        Significance (by zero-crossing) of left singular vectors
    V_sig : (J x N_COMP) ndarray
        Significance (by zero-crossing) of right singular vectors
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

    def run_perms(self):
        if len(self.U) < len(self.V): orig = self.U
        else: orig = self.V

        perms = compute.serial_permute(self.brain, self.behav,
                                       orig, self.d,
                                       n_perm=self._n_perm,
                                       n_split=self._n_split)
        self.d_pvals = compute.perm_sig(perms, self.d)

    def run_boots(self):
        U_boot, V_boot = compute.bootstrap(self.brain, self.behav,
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
