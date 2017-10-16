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
        and `G` is an optional grouping factor
    behav : (N x J [x G]) array_like
        Where `N` is the number of subjects, `J` is the number of observations,
        and `G` is an optional grouping factor
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during permutation testing. Default:
        None
    p : float (0,1), optional
        Signifiance criterion for bootstrapping, within (0, 1). Default: 0.05
    verbose : bool, optional
        Whether to print status updates. Default: True
    n_proc : int, optional
        If not None, number of processes to use for multiprocessing permutation
        testing/bootstrapping. Default: None
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Attributes
    -------
    U : (K[*G] x L) ndarray
        Left singular vectors
    d : (L x N_COM) ndarray
        Diagonal matrix of singular values
    V : (J x L) ndarray
        Right singular vectors
    ucorr, vcorr : (L,) ndarray
        Correlations of split-half resamples of singular vectors. Only present
        if n_split was specified at instantiation.
    u_pvals, v_pvals : (L,) ndarray
        P-values of singular vectors as determined by split-half resampling.
        Only present if n_split was specified at instantiation.
    d_pvals : (L,) ndarray
        P-values of latent variables as determined by permutation testing. Only
        present if n_split was NOT specified at instantiation.
    d_kaiser : (L,) ndarray
        Relevance of latent variables as determined by Kaiser criterion
    d_varexp : (L,) ndarray
        Percent variance explained by each latent variable
    U_bci : (K[*G] x L x 2) ndarray
        Bootstrapped CI for left singular vectors
    V_bci : (J x L x 2) ndarray
        Bootstrapped CI for right singular vectors
    U_bsr : (K[*G] x L) ndarray
        Bootstrap ratios for left singular vectors
    V_bsr : (J x L) ndarray
        Bootstrap ratios for right singular vectors
    U_sig : (K[*G] x L) ndarray
        Significance (by zero-crossing) of left singular vectors
    V_sig : (J x L) ndarray
        Significance (by zero-crossing) of right singular vectors

    References
    ----------
    .. [1] McIntosh, A. R., Bookstein, F. L., Haxby, J. V., & Grady, C. L.
       (1996). Spatial pattern analysis of functional brain images using
       partial least squares. Neuroimage, 3(3), 143-157.
    .. [2] McIntosh, A. R., & Lobaugh, N. J. (2004). Partial least squares
       analysis of neuroimaging data: applications and advances. Neuroimage,
       23, S250-S263.
    .. [3] Krishnan, A., Williams, L. J., McIntosh, A. R., & Abdi, H. (2011).
       Partial Least Squares (PLS) methods for neuroimaging: a tutorial and
       review. Neuroimage, 56(2), 455-475.
    .. [4] Kovacevic, N., Abdi, H., Beaton, D., & McIntosh, A. R. (2013).
       Revisiting PLS resampling: comparing significance versus reliability
       across range of simulations. In New Perspectives in Partial Least
       Squares and Related Methods (pp. 159-170). Springer, New York, NY.
       Chicago
    """

    def __init__(self, brain, behav,
                 n_perm=5000, n_boot=1000, n_split=None,
                 p=0.05,
                 verbose=True,
                 n_proc=None,
                 seed=None):
        self.brain, self.behav = brain, behav
        self._n_perm, self._n_boot, self._n_split = n_perm, n_boot, n_split
        self._n_proc = n_proc
        self._p = p

        if seed is not None: np.random.seed(seed)

        self.run_svd()
        self.run_perms(verbose=verbose)
        self.run_boots(verbose=verbose)
        self.get_sig()

    def run_svd(self):
        self.U, self.d, self.V = compute.svd(self.brain,
                                             self.behav)
        if self._n_split is not None:
            self.ucorr, self.vcorr = compute.split_half(self.brain,
                                                        self.behav,
                                                        n_split=self._n_split)

    def run_perms(self, verbose=True):
        if len(self.U) < len(self.V): orig = self.U
        else: orig = self.V

        if self._n_proc is not None:
            perms = compute.parallel_permute(self.brain, self.behav, orig,
                                             n_perm=self._n_perm,
                                             n_split=self._n_split,
                                             n_proc=self._n_proc)
        else:
            perms = compute.serial_permute(self.brain, self.behav, orig,
                                           n_perm=self._n_perm,
                                           n_split=self._n_split,
                                           verbose=verbose)

        if self._n_split is not None:
            self.u_pvals = compute.perm_sig(perms[:, :, 0],
                                            np.diag(self.ucorr))
            self.v_pvals = compute.perm_sig(perms[:, :, 1],
                                            np.diag(self.vcorr))
        else:
            self.d_pvals = compute.perm_sig(perms, self.d)

    def run_boots(self, verbose=True):
        U_boot, V_boot = compute.bootstrap(self.brain, self.behav,
                                           self.U, self.V,
                                           n_boot=self._n_boot,
                                           verbose=verbose)
        self.U_bci, self.V_bci = compute.boot_ci(U_boot, V_boot, p=self._p)
        self.U_bsr, self.V_bsr = compute.boot_rel(self.U, self.V,
                                                  U_boot, V_boot)

    def get_sig(self):
        self.U_sig = compute.boot_sig(self.U_bci)
        self.V_sig = compute.boot_sig(self.V_bci)

        self.d_kaiser = compute.kaiser_criterion(self.d)
        self.d_varexp = compute.crossblock_cov(self.d)
