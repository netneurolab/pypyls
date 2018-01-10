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
        self._run_pls(brain, behav, grouping=grouping)

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

    def _gen_bootsamp(self, X, Y, grouping=None, n_boot=500, seed=None):
        """
        Generates bootstrap resample arrays to be used in ``self._bootstrap``

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : (N,) array_like, optional
        n_boot : int, optional
            Number of boostraps to run. Default: 500

        Returns
        -------
        bootsamp : (N x B) np.ndarray
        """

        rs = utils.get_seed(seed)
        bootsamp = np.zeros(shape=(len(X), n_boot), dtype=int)

        for i in range(n_boot):
            if grouping is not None:
                inds = np.zeros(shape=len(grouping))
                for grp in np.unique(grouping):
                    curr_group = np.argwhere(grouping == grp).squeeze()
                    inds[curr_group] = rs.choice(curr_group,
                                                 size=len(curr_group),
                                                 replace=True)
            else:
                inds = rs.choice(np.arange(len(X)), size=len(X), replace=True)
            bootsamp[i] = inds

        return bootsamp


    def _run_pls(self, X, Y, grouping=None):
        """
        Runs PLS analysis
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(X, Y, grouping=grouping,
                                           seed=self._rs)
        # brain / design scores
        self.usc, self.vsc = X @ self.V, Y @ self.U

        # compute permutations
        perms = self._permutation(X, Y, grouping=grouping)

        # get split half reliability, if set
        if self.n_split is not None:
            self.ucorr, self.vcorr = self._split_half(X, Y, grouping=grouping,
                                                      seed=self._rs)
            self.u_pvals = utils.perm_sig(perms[:, :, 0], np.diag(self.ucorr))
            self.v_pvals = utils.perm_sig(perms[:, :, 1], np.diag(self.vcorr))
        else:
            self.d_pvals = utils.perm_sig(perms, self.d)

        # compute bootstraps
        U_boot, V_boot = self._bootstrap(X, Y, grouping=grouping)

        self.U_bci, self.V_bci = utils.boot_ci(U_boot, V_boot, ci=self.ci)
        self.U_bsr, self.V_bsr = utils.boot_rel(self.U @ self.d,
                                                self.V @ self.d,
                                                U_boot, V_boot)

        self.U_sig = utils.boot_sig(self.U_bci)
        self.V_sig = utils.boot_sig(self.V_bci)

        self.d_kaiser = utils.kaiser_criterion(self.d)
        self.d_varexp = utils.crossblock_cov(self.d)


class MeanCenteredPLS(BasePLS):
    """
    Runs PLS on `data` and `groups` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    mean-centered matrix generated from `data`

    Parameters
    ----------
    data : (N x K) array_like
        Where `N` is the number of subjects and  `K` is the number of
        observations
    groups : (N x J) array_like
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

    def __init__(self, data, groups, **kwargs):
        super(MeanCenteredPLS, self).__init__(**kwargs)
        self.data, self.groups = data, groups
        self._run_pls(data, utils.dummy_code(groups))

    def _svd(self, X, Y, seed=None, grouping=None):
        """
        Runs SVD on a mean-centered matrix computed from ``X`` and ``Y``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        grouping : placeholder
            Here for compatibility purposes; does nothing.

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
                                 random_state=utils.get_seed(seed))

        return U, np.diag(d), V.T

    def _gen_bootsamp(self, X, Y, grouping=None):
        """
        Generates bootstrap resample arrays to be used in ``self._bootstrap()``

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : placeholder

        Returns
        -------
        bootsamp : (N x B) np.ndarray
        """

        bootsamp = np.zeros(shape=(len(X), self.n_boot), dtype=int)
        grouping = utils.reverse_dummy_code(Y)

        for i in range(self.n_boot):
            inds = np.zeros(shape=len(grouping), dtype=int)
            for grp in np.unique(grouping):
                curr_group = np.argwhere(grouping == grp).squeeze()
                inds[curr_group] = self._rs.choice(curr_group,
                                                   size=len(curr_group),
                                                   replace=True)
            bootsamp[:, i] = inds

        return bootsamp

    def _gen_permsamp(self, X, Y, grouping=None):
        """
        Generates permutation arrays to be used in ``self.permutation()``

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : placeholder

        Returns
        -------
        permsamp : (N x B) np.ndarray
        """

        permsamp = np.zeros(shape=(len(X), self.n_perm), dtype=int)
        grouping = utils.dummy_code(Y)
        orig_gm = get_group_mean(X, grouping).mean()

        # ensure that all permutations are different from original grouping
        for i in range(self.n_perm):
            perm = self._rs.permutation(np.arange(len(X), dtype=int))
            while get_group_mean(X[perm], grouping).mean() == orig_gm:
                perm = self._rs.permutation(perm)
            permsamp[:, i] = perm

        return permsamp

    def _run_pls(self, X, Y):
        """
        Runs PLS analysis
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(X, Y, seed=self._rs)
        # brain / design scores
        self.usc, self.vsc = X @ self.V, Y @ self.U

        # compute permutations
        perms = self._permutation(X, Y)
        # get split half reliability, if set
        if self.n_split is not None:
            self.ucorr, self.vcorr = self._split_half(X, Y, seed=self._rs)
            self.u_pvals = utils.perm_sig(perms[:, :, 0], np.diag(self.ucorr))
            self.v_pvals = utils.perm_sig(perms[:, :, 1], np.diag(self.vcorr))
        else:
            self.d_pvals = utils.perm_sig(perms, self.d)

        # compute bootstraps
        U_boot, V_boot = self._bootstrap(X, Y)

        self.usc2 = get_mean_norm(X, Y) @ self.V
        self.orig_usc = get_group_mean(self.usc2, Y, grand=False)

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
    Parameters
    ----------
    X : (N x K) array_like
    Y : (N x G) array_like
        Dummy coded group array
    grand : bool, optional
        Default : True

    Returns
    -------
    group_mean : {(G,) or (G x K)} np.ndarray
        If grand is set, returns array with shape (G,); else, returns (G x K)
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
    Parameters
    ----------
    X : (N x K) array_like
    Y : (N x G) array_like
        Dummy coded group array

    Returns
    -------
    X_mean_centered : (N x K) np.ndarray
        ``X`` centered based on grand mean (i.e., mean of group means)
    """

    grand_mean = get_group_mean(X, Y)
    X_mean_centered = np.zeros_like(X)

    for grp in Y.T.astype('bool'):
        X_mean_centered[grp] = X[grp] - grand_mean

    return X_mean_centered
