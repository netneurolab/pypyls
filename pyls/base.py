# -*- coding: utf-8 -*-

import gc
import warnings
import numpy as np
from sklearn.utils.validation import check_random_state
from . import compute, structures, utils


def gen_permsamp(groups, n_cond, n_perm, seed=None, verbose=True):
    """
    Generates permutation arrays for PLS permutation testing

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of `G` groups
    n_cond : int
        Number of conditions, for each subject. Default: 1
    n_perm : int
        Number of permutations for which to generate resampling arrays
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print status updates as permutations are generated.
        Default: True

    Returns
    -------
    permsamp : (S, P) `numpy.ndarray`
        Subject permutation arrays, where `S` is the number of subjects and `P`
        is the requested number of permutations (i.e., `P = n_perm`)
    """
    Y = utils.dummy_code(groups, n_cond)
    permsamp = np.zeros(shape=(len(Y), n_perm), dtype=int)
    subj_inds = np.arange(np.sum(groups), dtype=int)
    rs = check_random_state(seed)
    warned = False

    # calculate some variables for permuting conditions within subject
    # do this here to save on calculation time
    indices, grps = np.where(Y)
    grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
    to_permute = [np.vstack(grp_conds[i:i + n_cond]) for i in
                  range(0, Y.shape[-1], n_cond)]
    splitinds = np.cumsum(groups)[:-1]
    check_grps = utils.dummy_code(groups).T.astype(bool)

    for i in utils.trange(n_perm, verbose=verbose, desc='Making permutations'):
        count, duplicated = 0, True
        while duplicated and count < 500:
            count, duplicated = count + 1, False
            # generate conditions permuted w/i subject
            inds = np.hstack([utils.permute_cols(i, seed=rs) for i
                              in to_permute])
            # generate permutation of subjects across groups
            perm = rs.permutation(subj_inds)
            # confirm subjects *are* mixed across groups
            if len(groups) > 1:
                for grp in check_grps:
                    if np.all(np.sort(perm[grp]) == subj_inds[grp]):
                        duplicated = True
            # permute conditions w/i subjects across groups and stack
            perminds = np.hstack([f.flatten('F') for f in
                                  np.split(inds[:, perm].T, splitinds)])
            # make sure permuted indices are not a duplicate sequence
            dupe_seq = perminds[:, None] == permsamp[:, :i]
            if dupe_seq.all(axis=0).any():
                duplicated = True
        # if we broke out because we tried 500 permutations and couldn't
        # generate a new one, just warn that we're using duplicate
        # permutations and give up
        if count == 500 and not warned:
            warnings.warn('WARNING: Duplicate permutations used.')
            warned = True
        # store the permuted indices
        permsamp[:, i] = perminds

    return permsamp


def gen_bootsamp(groups, n_cond, n_boot, seed=None, verbose=True):
    """
    Generates bootstrap arrays for PLS bootstrap resampling

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of `G` groups
    n_cond : int
        Number of conditions, for each subject. Default: 1
    n_boot : int
        Number of boostraps for which to generate resampling arrays
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print status updates as bootstrap samples are genereated.
        Default: True

    Returns
    -------
    bootsamp : (S, B) `numpy.ndarray`
        Subject bootstrap arrays, where `S` is the number of subjects and `B`
        is the requested number of bootstraps (i.e., `B = n_boot`)
    """
    Y = utils.dummy_code(groups, n_cond)
    bootsamp = np.zeros(shape=(len(Y), n_boot), dtype=int)
    subj_inds = np.arange(np.sum(groups), dtype=int)
    rs = check_random_state(seed)
    warned = False
    min_subj = int(np.ceil(Y.sum(axis=0).min() * 0.5))

    # calculate some variables for ensuring we resample with replacement
    # subjects across all their conditions. do this here to save on
    # calculation time
    indices, grps = np.where(Y)
    grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
    inds = np.hstack([np.vstack(grp_conds[i:i + n_cond]) for i
                      in range(0, len(grp_conds), n_cond)])
    splitinds = np.cumsum(groups)[:-1]
    check_grps = utils.dummy_code(groups).T.astype(bool)

    for i in utils.trange(n_boot, verbose=verbose, desc='Making bootstraps'):
        count, duplicated = 0, True
        while duplicated and count < 500:
            count, duplicated = count + 1, False
            # empty container to store current bootstrap attempt
            boot = np.zeros(shape=(subj_inds.size), dtype=int)
            # iterate through and resample from w/i groups
            for grp in check_grps:
                curr_grp, all_same = subj_inds[grp], True
                while all_same:
                    num_subj = curr_grp.size
                    boot[curr_grp] = np.sort(rs.choice(curr_grp,
                                                       size=num_subj,
                                                       replace=True),
                                             axis=0)
                    # make sure bootstrap has enough unique subjs
                    if np.unique(boot[curr_grp]).size >= min_subj:
                        all_same = False
            # resample subjects (with conditions) and stack groups
            bootinds = np.hstack([f.flatten('F') for f in
                                  np.split(inds[:, boot].T, splitinds)])
            # make sure bootstrap is not a duplicated sequence
            for grp in check_grps:
                curr_grp = subj_inds[grp]
                check = bootinds[curr_grp, None] == bootsamp[curr_grp, :i]
                if check.all(axis=0).any():
                    duplicated = True
        # if we broke out because we tried 500 bootstraps and couldn't
        # generate a new one, just warn that we're using duplicate
        # bootstraps and give up
        if count == 500 and not warned:
            warnings.warn('WARNING: Duplicate bootstraps used.')
            warned = True
        # store the bootstrapped indices
        bootsamp[:, i] = bootinds

    return bootsamp


def gen_splits(groups, n_cond, n_split, seed=None, test_size=0.5):
    """
    Generates splitting arrays for PLS split-half resampling and CV

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of `G` groups
    n_cond : int
        Number of conditions, for each subject. Default: 1
    n_split : int
        Number of splits for which to generate resampling arrays
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None
    test_size : (0, 1) float, optional
        Percent of subjects to include in the split halves. Default: 0.5

    Returns
    -------
    splitsamp : (S, I) `numpy.ndarray`
        Subject split arrays, where `S` is the number of subjects and `I`
        is the requested number of splits (i.e., `I = n_split`)
    """
    Y = utils.dummy_code(groups, n_cond)
    splitsamp = np.zeros(shape=(len(Y), n_split), dtype=bool)
    subj_inds = np.arange(np.sum(groups), dtype=int)
    rs = check_random_state(seed)
    warned = False

    # calculate some variables for permuting conditions within subject
    # do this here to save on calculation time
    indices, grps = np.where(Y)
    grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
    inds = np.hstack([np.vstack(grp_conds[i:i + n_cond]) for i
                      in range(0, len(grp_conds), n_cond)])
    splitinds = np.cumsum(groups)[:-1]
    check_grps = utils.dummy_code(groups).T.astype(bool)

    for i in range(n_split):
        count, duplicated = 0, True
        while duplicated and count < 500:
            count, duplicated = count + 1, False
            # empty containter to store current split half attempt
            split = np.zeros(shape=(subj_inds.size), dtype=bool)
            # iterate through and split each group separately
            for grp in check_grps:
                curr_grp = subj_inds[grp]
                take = rs.choice([np.ceil, np.floor])
                num_subj = int(take(curr_grp.size * (1 - test_size)))
                splinds = rs.choice(curr_grp,
                                    size=num_subj,
                                    replace=False)
                split[splinds] = True
            # split subjects (with conditions) and stack groups
            half = np.hstack([f.flatten('F') for f in
                              np.split(((inds + 1).astype(bool)
                                        * [split[None]]).T,
                                       splitinds)])
            # make sure split half is not a duplicated sequence
            dupe_seq = half[:, None] == splitsamp[:, :i]
            if dupe_seq.all(axis=0).any():
                duplicated = True
        if count == 500 and not warned:
            warnings.warn('WARNING: Duplicate split halves used.')
            warned = True
        splitsamp[:, i] = half

    return splitsamp


class BasePLS():
    """
    Base PLS class to be subclassed

    Contains most of the math required for PLS, leaving a few functions for PLS
    subclasses to implement. This will not run without those implementations.

    Parameters
    ----------
    {input_matrix}
    {groups}
    {conditions}
    **kwargs : optional
        Additional key-value pairs; see :obj:`pyls.structures.PLSInputs` for
        more info

    References
    ----------

    {references}
    """.format(**structures._pls_input_docs)

    def __init__(self, X, Y=None, groups=None, n_cond=1, **kwargs):
        # if groups aren't provided or are provided wrong, fix them
        if groups is None:
            groups = [len(X) // n_cond]
        elif not isinstance(groups, (list, np.ndarray)):
            groups = [groups]

        # coerce groups to integers
        groups = [int(g) for g in groups]

        # check that data matrices and groups + n_cond inputs jibe
        n_samples = sum([g * n_cond for g in groups])
        if len(X) != n_samples:
            raise ValueError('Number of samples specified by `groups` and '
                             '`n_cond` does not match number of samples in '
                             'input array(s).\n'
                             '    EXPECTED: {}\n'
                             '    ACTUAL:   {} (groups: {} * n_cond: {})'
                             .format(len(X), n_samples, groups, n_cond))

        if Y is not None and len(X) != len(Y):
            raise ValueError('Provided `X` and `Y` matrices must have the '
                             'same number of samples. Provided matrices '
                             'differed: X: {}, Y: {}'.format(len(X), len(Y)))

        self.inputs = structures.PLSInputs(X=X, Y=Y, groups=groups,
                                           n_cond=n_cond, **kwargs)
        # store dummy-coded array of groups / conditions (save on computation)
        self.dummy = utils.dummy_code(groups, n_cond)
        self.rs = check_random_state(self.inputs.get('seed'))

        # check for parallel processing desire
        n_proc = self.inputs.get('n_proc')
        if n_proc is not None and n_proc != 1 and not utils.joblib_avail:
            self.inputs.n_proc = None
            warnings.warn('Setting n_proc > 1 requires the joblib module. '
                          'Considering installing joblib and re-running this '
                          'if you would like parallelization. Resetting '
                          'n_proc to 1 for now.')

    def gen_covcorr(self, X, Y, groups=None):
        """
        Should generate cross-covariance array to be used in `self._svd()`

        Must accept the listed parameters and return one array

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        groups : (G,) array_like
            Array with number of subjects in each of `G` groups

        Returns
        -------
        crosscov : np.ndarray
            Covariance array for decomposition
        """

        raise NotImplementedError

    def gen_distrib(self, X, Y, groups=None, original=None):
        """
        Should generate behavioral correlations or contrast for bootstrap

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        groups : (S, J) array_like
            Dummy coded array, where `S` is observations and `J` corresponds to
            the number of different groups x conditions represented in `X` and
            `Y`. A value of 1 indicates that an observation belongs to a
            specific group or condition

        Returns
        -------
        distrib : (T, L)
            Behavioral correlations or contrast for single bootstrap resample
        """

        raise NotImplementedError

    def run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features

        Returns
        -------
        results : :obj:`pyls.structures.PLSResults`
            Results of PLS (not including PLS type-specific outputs)
        """

        # initate results structure
        self.res = res = structures.PLSResults(inputs=self.inputs)

        # get original singular vectors / values
        res['x_weights'], res['singvals'], res['y_weights'] = \
            self.svd(X, Y, seed=self.rs)
        res['x_scores'] = X @ res['x_weights']

        if self.inputs.n_perm > 0:
            # compute permutations and get statistical significance of LVs
            d_perm, ucorrs, vcorrs = self.permutation(X, Y, seed=self.rs)
            res['permres']['pvals'] = compute.perm_sig(res['singvals'], d_perm)
            res['permres']['permsamples'] = self.permsamp
            res['permres']['perm_singval'] = d_perm

            if self.inputs.n_split is not None:
                # get ucorr / vcorr (via split half resampling) for original,
                # unpermuted `X` and `Y` arrays
                di = np.linalg.inv(res['singvals'])
                orig_ucorr, orig_vcorr = self.split_half(X, Y,
                                                         res['x_weights'] @ di,
                                                         res['y_weights'] @ di,
                                                         seed=self.rs)
                # get p-values for ucorr/vcorr
                ucorr_prob = compute.perm_sig(np.diag(orig_ucorr), ucorrs)
                vcorr_prob = compute.perm_sig(np.diag(orig_vcorr), vcorrs)

                # get confidence intervals for ucorr/vcorr
                ucorr_ll, ucorr_ul = compute.boot_ci(ucorrs, ci=self.inputs.ci)
                vcorr_ll, vcorr_ul = compute.boot_ci(vcorrs, ci=self.inputs.ci)

                # update results object with split-half resampling results
                res['splitres'].update(dict(ucorr=orig_ucorr,
                                            vcorr=orig_vcorr,
                                            ucorr_pvals=ucorr_prob,
                                            vcorr_pvals=vcorr_prob,
                                            ucorr_lolim=ucorr_ll,
                                            vcorr_lolim=vcorr_ll,
                                            ucorr_uplim=ucorr_ul,
                                            vcorr_uplim=vcorr_ul))

        return res

    def svd(self, X, Y, groups=None, seed=None):
        """
        Runs SVD on cross-covariance matrix computed from `X` and `Y`

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        groups : (S, J) array_like
            Dummy coded array, where `S` is observations and `J` corresponds to
            the number of different groups x conditions represented in `X` and
            `Y`. A value of 1 indicates that an observation belongs to a
            specific group or condition
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        U : (B, L) `numpy.ndarray`
            Left singular vectors from singular value decomposition
        d : (L, L) `numpy.ndarray`
            Diagonal array of singular values from singular value decomposition
        V : (J, L) `numpy.ndarray`
            Right singular vectors from singular value decomposition
        """

        # make dummy-coded grouping array if not provided
        if groups is None:
            groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)

        # generate cross-covariance matrix and determine # of components
        crosscov = self.gen_covcorr(X, Y, groups=groups)
        U, d, V = compute.svd(crosscov, seed=seed)

        return U, d, V

    def bootstrap(self, X, Y, seed=None):
        """
        Bootstraps `X` and `Y` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        distrib : (T, L) numpy.ndarray
            Either behavioral correlations or group x condition contrast;
            depends on PLS type
        u_sum : (B, L) numpy.ndarray
            Sum of the left singular vectors across all bootstraps
        u_square : (B, L) numpy.ndarray
            Sum of the squared left singular vectors across all bootstraps
        """

        # generate bootstrap resampled indices (unless already provided)
        self.bootsamp = self.inputs.get('bootsamples', None)
        if self.bootsamp is None:
            self.bootsamp = gen_bootsamp(self.inputs.groups,
                                         self.inputs.n_cond,
                                         self.inputs.n_boot,
                                         seed=seed,
                                         verbose=self.inputs.verbose)

        # make empty arrays to store bootstrapped singular vectors
        # these will be used to calculate the standard error later on for
        # creation of bootstrap ratios
        u_sum = np.zeros_like(self.res['x_weights'])
        u_square = np.zeros_like(self.res['x_weights'])

        # `distrib` corresponds either to the behavioral correlations (if
        # running a behavioral PLS) or to the group/condition contrast (if
        # running a mean-centered PLS); we'll just extend it and then stack
        # all the individual matrices together later (they're quite small so we
        # don't need to be too worried about memory usage, here)
        distrib = []

        # determine the number of bootstraps we'll run each iteration
        iters = 1 if self.inputs.n_proc is None else self.inputs.n_proc
        gen = utils.trange(self.inputs.n_boot, verbose=self.inputs.verbose,
                           desc='Running bootstraps')

        with utils.get_par_func(self.inputs.n_proc,
                                self.__class__._single_boot) as (par, func):
            boots = 0
            while boots < self.inputs.n_boot:
                # determine number of bootstraps to run this round
                # we don't want to overshoot the requested number, so make
                # sure to cut it off if that's what wold happen
                top = boots + iters
                if top >= self.inputs.n_boot:
                    top = self.inputs.n_boot

                # run the bootstraps
                d, usu = zip(*par(func(self, X=X, Y=Y,
                                       inds=self.bootsamp[..., i],
                                       groups=self.dummy,
                                       original=self.res['x_weights'],
                                       seed=i)
                                  for i in range(boots, top)))

                # sum bootstrapped singular vectors and store
                u_sum += np.sum(usu, axis=0)
                u_square += np.sum(np.square(usu), axis=0)
                distrib.extend(d)

                # force garbage collection
                # this is only really needed when parallelizing bootstraps
                # the `usu` variable can get REALLY GIANT if either `X` or `Y`
                # is large and `n_proc` is > 1, so we really don't want to keep
                # it around for any longer than absolutely necessary
                if self.inputs.n_proc is not None:
                    del usu
                    gc.collect()

                # update progress bar and # of bootstraps already run
                gen.update(top - boots)
                boots = top
        gen.close()

        return np.stack(distrib, axis=-1), u_sum, u_square

    def _single_boot(self, X, Y, inds, groups=None, original=None, seed=None):
        """
        Bootstraps `X` and `Y` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        groups : (S, J) array_like
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        original : (B, L) array_like
            Left singular vector from original decomposition of `X` and `Y`.
            Used to perform Procrustes rotation on permuted singular vectors
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        distrib : np.ndarray
            Either behavioral correlations or contrast, depending on PLS type;
            generated with self.gen_distrib() which should be specified by the
            PLS subclass
        U_sum : (B, L) array_like
            Left singular vectors from decomposition of bootstrap resampled `X`
            and `Y`
        """

        # make sure we have original (non-bootstrapped) singular vectors
        # these are required for the procrustes rotation to ensure our
        # singular vectors are all in the same orientation
        if original is None:
            original = self.svd(X, Y, groups=groups, seed=seed)[0]

        # perform SVD of bootstrapped arrays and rotate left singular vectors
        U, d = self.svd(X[inds], Y[inds], groups=groups, seed=seed)[:-1]
        U_boot = compute.procrustes(original, U, d)

        # get contrast / behavcorrs (this function should be specified by the
        # subclass)
        distrib = self.gen_distrib(X[inds], Y[inds], original, groups)

        return distrib, U_boot

    def make_permutation(self, X, Y, perminds):
        """
        Permutes `Y` according to `perminds`, leaving `X` un-permuted

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        perminds : (S,) array_like
            Array by which to permute `Y`

        Returns
        -------
        Xp : (S, B) array_like
            Identical to `X`
        Yp : (S, T) array_like
            `Y`, permuted according to `perminds`
        """

        return X, Y[perminds]

    def permutation(self, X, Y, seed=None):
        """
        Permutes `X` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        d_perm : (L, P) `numpy.ndarray`
            Permuted singular values, where `L` is the number of singular
            values and `P` is the number of permutations
        ucorrs : (L, P) `numpy.ndarray`
            Split-half correlations of left singular values. Only set if
            `self.inputs.n_split != 0`
        vcorrs : (L, P) `numpy.ndarray`
            Split-half correlations of right singular values. Only set if
            `self.inputs.n_split != 0`
        """

        # generate permuted indices (unless already provided)
        self.permsamp = self.inputs.get('permsamples')
        if self.permsamp is None:
            self.permsamp = gen_permsamp(self.inputs.groups,
                                         self.inputs.n_cond,
                                         self.inputs.n_perm,
                                         seed=seed,
                                         verbose=self.inputs.verbose)

        # get permuted values (parallelizing as requested)
        gen = utils.trange(self.inputs.n_perm, verbose=self.inputs.verbose,
                           desc='Running permutations')
        with utils.get_par_func(self.inputs.n_proc,
                                self.__class__._single_perm) as (par, func):
            out = par(func(self, X=X, Y=Y, inds=self.permsamp[:, i],
                           groups=self.dummy, original=self.res['y_weights'],
                           seed=i)
                      for i in gen)
        d_perm, ucorrs, vcorrs = [np.stack(o, axis=-1) for o in zip(*out)]

        return d_perm, ucorrs, vcorrs

    def _single_perm(self, X, Y, inds, groups=None, original=None, seed=None):
        """
        Permutes `X` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        inds : (S,) array_like
            Permutation resampling array
        original : (J, L) array_like
            Right singular vector from original decomposition of `X` and `Y`.
            Used to perform Procrustes rotation on permuted singular values,
            if desired
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        d_perm : (L,) `numpy.ndarray`
            Permuted singular values, where `L` is the number of singular
            values
        ucorrs : (L,) `numpy.ndarray`
            Split-half correlations of left singular values. Only set if
            `self.inputs.n_split != 0`
        vcorrs : (L,) `numpy.ndarray`
            Split-half correlations of right singular values. Only set if
            `self.inputs.n_split != 0`
        """

        # calculate SVD of permuted matrices
        Xp, Yp = self.make_permutation(X, Y, inds)
        U, d, V = self.svd(Xp, Yp, groups=groups, seed=seed)

        # optionally get rotated/rescaled singular values
        if self.inputs.rotate:
            if original is None:
                original = self.svd(X, Y, groups=groups, seed=seed)[-1]
            ssd = np.sqrt(np.sum(compute.procrustes(original, V, d)**2,
                                 axis=0))
        else:
            ssd = np.diag(d)

        # get ucorr/vcorr if split-half resampling requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(d)
            ucorr, vcorr = self.split_half(Xp, Yp, U @ di, V @ di,
                                           groups=groups, seed=seed)
        else:
            ucorr, vcorr = None, None

        return ssd, ucorr, vcorr

    def split_half(self, X, Y, ud=None, vd=None, groups=None, seed=None):
        """
        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        ud : (B, L) array_like
            Left singular vectors, scaled by singular values
        vd : (J, L) array_like
            Right singular vectors, scaled by singular values
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        ucorr : (L,) `numpy.ndarray`
            Average correlation of left singular vectors across split-halves
        vcorr : (L,) `numpy.ndarray`
            Average correlation of right singular vectors across split-halves
        """

        # generate splits
        splitsamp = gen_splits(self.inputs.groups,
                               self.inputs.n_cond,
                               self.inputs.n_split,
                               seed=seed,
                               test_size=0.5).astype(bool)

        # make dummy-coded grouping array if not provided
        if groups is None:
            groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)

        # generate original singular vectors if not provided
        if ud is None or vd is None:
            U, d, V = self.svd(X, Y, groups=groups, seed=seed)
            di = np.linalg.inv(d)
            ud, vd = U @ di, V @ di

        # empty arrays to hold split-half correlations
        ucorr = np.zeros(shape=(ud.shape[-1], self.inputs.n_split))
        vcorr = np.zeros(shape=(vd.shape[-1], self.inputs.n_split))

        for i in range(self.inputs.n_split):
            # calculate cross-covariance matrix for both splits
            spl = splitsamp[:, i]
            D1 = self.gen_covcorr(X[spl], Y[spl], groups=groups[spl])
            D2 = self.gen_covcorr(X[~spl], Y[~spl], groups=groups[~spl])

            # project cross-covariance matrices onto original SVD to obtain
            # left & right singular vector and correlate between split halves
            ucorr[:, i] = compute.efficient_corr(D1.T @ vd, D2.T @ vd)
            vcorr[:, i] = compute.efficient_corr(D1 @ ud, D2 @ ud)

        # return average correlations for singular vectors across `n_split`
        return np.mean(ucorr, axis=-1), np.mean(vcorr, axis=-1)
