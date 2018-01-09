# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.extmath import randomized_svd
from pyls.base import BasePLS
from pyls import utils


class BehavioralPLS(BasePLS):
    """
    Runs PLS on `brain` and `behav` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    cross-covariance matrix of `brain` and `behav`.

    Parameters
    ----------
    X : (N x K) array_like
        Where ``N`` is the number of subjects and ``K`` is the number of
        observations
    Y : (N x J) array_like
        Where ``N`` is the number of subjects and ``J`` is the number of
        observations
    grouping : (N,) array_like
        Array with labels separating ``N`` subjects into ``G`` groups
    n_perm : int, optional
        Number of permutations for testing statistical significance of singular
        vectors. Default: 5000
    n_boot : int, optional
        Number of bootstraps for testing reliability of singular vectors.
        Default: 1000
    n_split : int, optional
        Number of split-half resamples for testing reliability of permutations.
        Default: 500
    ci : (0, 100) float, optional
        Confidence interval to calculate from bootstrapped distributions.
        Default: 95
    n_proc : int, optional
        Number of processes to use for parallelizing permutation and
        bootstrapping. Default: 1
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Attributes
    ----------
    U : (J[*G] x L) ndarray
        Left singular vectors
    d : (L x L) ndarray
        Diagonal matrix of singular values
    V : (K x L) ndarray
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
    U_bci : (J[*G] x L x 2) ndarray
        Bootstrapped CI for left singular vectors
    V_bci : (K x L x 2) ndarray
        Bootstrapped CI for right singular vectors
    U_bsr : (J[*G] x L) ndarray
        Bootstrap ratios for left singular vectors
    V_bsr : (K x L) ndarray
        Bootstrap ratios for right singular vectors
    U_sig : (J[*G] x L) ndarray
        Significance (by zero-crossing) of left singular vectors
    V_sig : (K x L) ndarray
        Significance (by zero-crossing) of right singular vectors
    """
    def __init__(self, brain, behav, grouping=None, **kwargs):
        super(BehavioralPLS, self).__init__(**kwargs)
        self.X, self.Y, self.groups = brain, behav, grouping

        self._run_svd()
        self._run_perms()
        self._run_boots()
        self._get_sig()

    def _svd(self, X, Y, grouping=None, seed=None):
        """
        Runs SVD on the cross-covariance matrix of ``X`` and ``Y``

        Finds ``L`` singular vectors, where ``L`` is the minimum of the
        dimensions of ``X`` and ``Y`` if ``grouping`` is not provided, or the
        number of unique values in ``grouping`` if provided.

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables
        grouping : (N,) array_like, optional
            Grouping array, where ``len(np.unique(grouping))`` is the number of
            distinct groups in ``X`` and ``Y``. Cross-covariance matrices are
            computed separately for each group and stacked prior to SVD.
            Default: None
        seed : {int, RandomState instance, None}, optional
            The seed of the pseudo random number generator to use when
            shuffling the data.  If int, ``seed`` is the seed used by the
            random number generator. If RandomState instance, ``seed`` is the
            random number generator. If None, the random number generator is
            the RandomState instance used by ``np.random``. Default: None

        Returns
        -------
        U : (J[*G] x L) ndarray
            Left singular vectors, where ``G`` is the number of unique values
            in ``grouping`` if provided
        d : (L x L) ndarray
            Diagonal array of singular values
        V : (K x L) ndarray
            Right singular vectors
        """

        if X.ndim != Y.ndim:
            raise ValueError('Number of dimensions of ``X`` and ``Y`` must '
                             'match.')
        if X.ndim != 2:
            raise ValueError('``X`` and ``Y`` must each have 2 dimensions.')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('The first dimension of ``X`` and ``Y`` must '
                             'match.')

        if grouping is None:
            n_comp = min(min(X.shape), min(Y.shape))
            crosscov = utils.xcorr(utils.normalize(X), utils.normalize(Y))
        else:
            groups = np.unique(grouping)
            n_comp = len(groups)
            crosscov = np.row_stack([utils.xcorr(utils.normalize(X[grouping == grp]),
                                                 utils.normalize(Y[grouping == grp]))
                                     for grp in groups])

        U, d, V = randomized_svd(crosscov, n_components=n_comp,
                                 random_state=utils.get_seed(seed))

        return U, np.diag(d), V.T

    def _run_svd(self):
        self.U, self.d, self.V = self._svd(self.X, self.Y, self.groups)
        if self.n_split is not None:
            self.ucorr, self.vcorr = self._split_half(self.X, self.Y,
                                                      grouping=self.groups,
                                                      n_split=self.n_split)


class MeanCenteredPLS(BasePLS):
    """
    Runs PLS on `data` and `grouping` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    mean-centered matrix generated from `data`

    Parameters
    ----------
    X : (N x K) array_like
        Where `N` is the number of subjects and  `K` is the number of
        observations
    grouping : (N x J) array_like
        Where `N` is the number of subjects, `J` is the number of groups.
        Should be a dummy coded matrix (i.e., 1 indicates group membership)
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
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None
    """

    def __init__(self, data, grouping, **kwargs):
        super(MeanCenteredPLS, self).__init__(**kwargs)
        self.X, self.Y = data, utils.dummy_code(grouping)
        self._run_pls()

    def _svd(self, X, Y, grouping=None, seed=None):
        """
        Runs SVD on a mean-centered matrix computed from ``X`` and ``Y``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 in a given
            row/column indicates that subject belongs to a given group.
        seed : {int, RandomState instance, None}, optional
            The seed of the pseudo random number generator to use when
            shuffling the data.  If int, ``seed`` is the seed used by the
            random number generator. If RandomState instance, ``seed`` is the
            random number generator. If None, the random number generator is
            the RandomState instance used by ``np.random``. Default: None

        Returns
        -------
        U : (J x J-1) ndarray
            Left singular vectors
        d : (J-1 x J-1) ndarray
            Diagonal array of singular values
        V : (K x J-1) ndarray
            Right singular vectors
        """

        iden = np.ones(shape=(len(Y), 1))
        grp_means = np.linalg.inv(np.diag((iden.T @ Y).flatten())) @ Y.T @ X
        num_group = len(grp_means)
        L = np.ones(shape=(num_group, 1))
        # effectively the same as M - M.mean(axis=0)...
        mean_centered = grp_means - (L @ (((1/num_group) * L.T) @ grp_means))
        U, d, V = randomized_svd(mean_centered,
                                 n_components=Y.shape[-1]-1,
                                 random_state=seed)

        return U, np.diag(d), V.T

    def _run_pls(self):
        """
        Runs PLS analysis
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(self.X, self.Y, seed=self._rs)
        # brain / design scores
        self.usc, self.vsc = self.X @ self.V, self.Y @ self.U

        # compute permutations
        perms = self._permutation(self.X, self.Y,
                                  n_perm=self.n_perm,
                                  n_split=self.n_split,
                                  n_proc=self._n_proc)
        # get split half reliability, if set
        if self.n_split is not None:
            self.ucorr, self.vcorr = self._split_half(self.X, self.Y,
                                                      n_split=self.n_split,
                                                      seed=self._rs)
            self.u_pvals = utils.perm_sig(perms[:, :, 0], np.diag(self.ucorr))
            self.v_pvals = utils.perm_sig(perms[:, :, 1], np.diag(self.vcorr))
        else:
            self.d_pvals = utils.perm_sig(perms, self.d)

        Xmn = get_mean_norm(self.X, self.Y)
        self.usc2 = Xmn @ self.V
        self.orig_usc = get_group_mean(self.usc2, self.Y, grand=False)

        U_boot, V_boot, bootsamp = self._bootstrap(self.X, self.Y,
                                                   n_boot=self.n_boot)
        self.bootsamp = bootsamp
        self.U_bci, self.V_bci = utils.boot_ci(U_boot, V_boot, ci=self.ci)
        self.U_bsr, self.V_bsr = utils.boot_rel(self.U @ self.d,
                                                self.V @ self.d,
                                                U_boot, V_boot)

        self.U_sig = utils.boot_sig(self.U_bci)
        self.V_sig = utils.boot_sig(self.V_bci)

        self.d_kaiser = utils.kaiser_criterion(self.d)
        self.d_varexp = utils.crossblock_cov(self.d)


def get_group_mean(X, Y, grand=True):
    """
    """

    group_mean = np.zeros((Y.shape[-1], X.shape[-1]))

    for n, grp in enumerate(Y.T.astype('bool')):
        group_mean[n] = X[grp].mean(axis=0)

    if grand:
        return group_mean.sum(axis=0) / Y.shape[-1]
    else:
        return group_mean


def get_mean_norm(X, Y):
    """
    """

    grand_mean = get_group_mean(X, Y)
    X_mean_centered = np.zeros_like(X)

    for grp in Y.T.astype('bool'):
        X_mean_centered[grp] = X[grp] - grand_mean

    return X_mean_centered
