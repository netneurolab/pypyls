# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.extmath import randomized_svd
from pyls.base import BasePLS
from pyls import compute, utils


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
            crosscov = [utils.xcorr(utils.normalize(X[grouping == grp]),
                                    utils.normalize(Y[grouping == grp]))
                        for grp in groups]
            crosscov = np.row_stack(crosscov)

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
            self.upvals = compute.perm_sig(perms[:, :, 0], np.diag(self.ucorr))
            self.vpvals = compute.perm_sig(perms[:, :, 1], np.diag(self.vcorr))
        else:
            self.d_pvals = compute.perm_sig(perms, self.d)

        # compute bootstraps
        U_boot, V_boot = self._bootstrap(X, Y, grouping=grouping)

        self.U_bci, self.V_bci = compute.boot_ci(U_boot, V_boot, ci=self.ci)
        self.U_bsr, self.V_bsr = compute.boot_rel(self.U @ self.d,
                                                  self.V @ self.d,
                                                  U_boot, V_boot)

        self.U_sig = compute.boot_sig(self.U_bci)
        self.V_sig = compute.boot_sig(self.V_bci)

        self.d_kaiser = compute.kaiser_criterion(self.d)
        self.d_varexp = compute.crossblock_cov(self.d)


class MeanCenteredPLS(BasePLS):
    """
    Runs PLS on `data` and `groups` arrays

    Uses singular value decomposition (SVD) to find latent variables from
    mean-centered matrix generated from `data`

    Parameters
    ----------
    data : (N x K) array_like
        Where ``N`` is the number of subjects and ``K`` is the number of
        observations
    groups : (N x J) array_like
        Where ``N`` is the number of subjects, ``J`` is the number of groups.
        Should be a dummy coded matrix (i.e., 1 indicates group membership)
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during each permutation. Default: None
    ci : (0, 100) float, optional
        Confidence interval to calculate from bootstrapped distributions.
        Default: 95
    n_proc : int, optional
        Number of processors to use for permutation and bootstrapping.
        Default: 1 (no multiprocessing)
    seed : int, optional
        Seed for random number generator. Default: None
    verbose : bool, optional
        Print status updates
    """

    def __init__(self, data, groups, **kwargs):
        super(MeanCenteredPLS, self).__init__(**kwargs)
        # for consistency, assign variables to X and Y
        self.inputs._X, self.inputs._Y = data, utils.dummy_code(groups)
        # run analysis
        self._run_pls(self.inputs.X, self.inputs.Y)

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

        bootsamp = np.zeros(shape=(len(X), self.inputs.n_boot), dtype=int)
        min_subj = int(np.ceil(Y.sum(axis=0).min() * 0.5))
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_boot, desc='Making bootstraps'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty container to store current bootstrap attempt
                boot = np.zeros(shape=(subj_inds.size, 1), dtype=int)
                count, duplicated = count + 1, False
                # iterate through and resample from w/i groups
                for grp in Y.T.astype(bool):
                    curr_grp, all_same = subj_inds[grp], True
                    while all_same:
                        boot[curr_grp, 0] = self._rs.choice(curr_grp,
                                                            size=curr_grp.size,
                                                            replace=True)
                        # make sure bootstrap has enough unique subjs
                        if np.unique(boot[curr_grp]).size >= min_subj:
                            all_same = False
                # make sure bootstrap is not a duplicated sequence
                dupe_seq = np.sort(boot) == np.sort(bootsamp[:, :i], axis=0)
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate boostraps used.')
            bootsamp[:, i] = boot.squeeze()

        return bootsamp

    def _gen_permsamp(self, X, Y, grouping=None):
        """
        Generates permutation arrays to be used in ``self._permutation()``

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : placeholder

        Returns
        -------
        permsamp : (N x P) np.ndarray
        """

        permsamp = np.zeros(shape=(len(X), self.inputs.n_perm), dtype=int)
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_perm, desc='Making permutations'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # initial permutation attempt
                perm = self._rs.permutation(subj_inds)
                count, duplicated = count + 1, False
                # iterate through groups and ensure that we aren't just
                # permuting subjects *within* any of the groups
                for grp in Y.T.astype(bool):
                    if np.all(np.sort(perm[grp]) == subj_inds[grp]):
                        duplicated = True
                # make sure permutation is not a duplicated sequence
                dupe_seq = perm[:, None] == permsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate permutations used.')
            permsamp[:, i] = perm

        return permsamp

    def _gen_splits(self, X, Y, grouping=None):
        """
        Generates split half arrays to be using in ``self._split_half()``

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : placeholder

        Returns
        -------
        splitsamp : (N x S) np.ndarray
        """

        splitsamp = np.zeros(shape=(len(X), self.inputs.n_split), dtype=bool)
        subj_inds = np.arange(len(X), dtype=int)

        for i in range(self.inputs.n_split):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty containter to store current split half attempt
                split = np.zeros(shape=(subj_inds.size, 1), dtype=bool)
                count, duplicated = count + 1, False
                # iterate through and split each group separately
                for grp in Y.T.astype(bool):
                    curr_grp = subj_inds[grp]
                    take = self._rs.choice([np.ceil, np.floor])
                    num_subj = int(take(np.sum(curr_grp.size)/2))
                    inds = self._rs.choice(curr_grp,
                                           size=num_subj,
                                           replace=False)
                    split[inds] = True
                # make sure split half is not a duplicated sequence
                dupe_seq = split == splitsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate split halves used.')
            splitsamp[:, i] = split.squeeze()

        return splitsamp

    def _boot_distrib(self, X, Y, V_boot):
        """
        Generates bootstrapped distribution
        """

        distrib = np.zeros(shape=(self.U.shape + (self.inputs.n_boot,)))
        normed_V_boot = utils.normalize(V_boot)

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot, V = self.bootsamp[:, i], normed_V_boot[:, :, i]
            usc = compute.get_mean_norm(X[boot], Y) @ V
            distrib[:, :, i] = compute.get_group_mean(usc, Y, grand=False)

        return distrib

    def _run_pls(self, X, Y):
        """
        Runs PLS analysis
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(X, Y, seed=self._rs)

        # compute permutations
        d_perm, ucorrs, vcorrs = self._permutation(X, Y)

        # get LV significance
        self.d_pvals = compute.perm_sig(self.d, d_perm)

        # get variance explained by latent variables
        self.d_varexp = compute.crossblock_cov(self.d)

        # get split half reliability, if requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(self.d)
            ud, vd = self.U @ di, self.V @ di
            self.U_corr, self.V_corr = self._split_half(X, Y, ud, vd)
            self.U_pvals = compute.perm_sig(np.diag(self.U_corr), ucorrs)
            self.V_pvals = compute.perm_sig(np.diag(self.V_corr), vcorrs)

        # generate bootstrapped singular vectors
        U_boot, V_boot = self._bootstrap(X, Y)

        # compute bootstrap ratios
        self.U_bsr = compute.boot_rel(self.U @ self.d, U_boot)
        self.V_bsr = compute.boot_rel(self.V @ self.d, V_boot)

        # get normalized brain scores and contrast
        self.usc = compute.get_mean_norm(X, Y) @ self.V
        self.orig_usc = compute.get_group_mean(self.usc, Y, grand=False)

        # generate distribution / confidence intervals for contrast
        self.distrib = self._boot_distrib(X, Y, V_boot)
        self.orig_usc_ll, self.orig_usc_ul = compute.boot_ci(self.distrib,
                                                             ci=self.inputs.ci)
