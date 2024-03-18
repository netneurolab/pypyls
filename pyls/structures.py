# -*- coding: utf-8 -*-
"""
Data structures to hold PLS inputs and results objects
"""

from multiprocessing import cpu_count
from textwrap import dedent
from .utils import ResDict

_pls_input_docs = dict(
    decomposition_narrative=dedent("""\
    The singular value decomposition generates mutually orthogonal latent
    variables (LVs), comprised of left and right singular vectors and a
    diagonal matrix of singular values. The `i`-th pair of singular vectors
    detail the contributions of individual input features to an overall,
    multivariate pattern (the `i`-th LV), and the singular values explain the
    amount of variance captured by that pattern.

    Statistical significance of the LVs is determined via permutation testing.
    Bootstrap resampling is used to examine the contribution and reliability of
    the input features to each LV. Split-half resampling can optionally be used
    to assess the reliability of the LVs. A cross-validated framework can
    optionally be used to examine how accurate the decomposition is when
    employed in a predictive framework.\
    """),
    input_matrix=dedent("""\
    X : (S, B) array_like
        Input data matrix, where `S` is samples and `B` is features\
    """),
    groups=dedent("""\
    groups : (G,) list of int
        List with the number of subjects present in each of `G` groups. Input
        data should be organized as subjects within groups (i.e., groups should
        be vertically stacked). If there is only one group this can be left
        blank.\
    """),
    conditions=dedent("""\
    n_cond : int
        Number of conditions observed in data. Note that all subjects must have
        the same number of conditions. If both conditions and groups are
        present then the input data should be organized as subjects within
        conditions within groups (i.e., g1c1s[1-S], g1c2s[1-S], g2c1s[1-S],
        g2c2s[1-S]).\
    """),
    mean_centering=dedent("""\
    mean_centering : {0, 1, 2}, optional
        Mean-centering method to use. This will determine how the mean-centered
        matrix is generated and what effects are "boosted" during the SVD.
        Default: 0\
    """),
    # perms / resampling / crossval
    stat_test=dedent("""\
    n_perm : int, optional
        Number of permutations to use for testing significance of components.
         Default: 5000
    n_boot : int, optional
        Number of bootstraps to use for testing reliability of data features.
        Default: 5000\
    """),
    split_half=dedent("""\
    n_split : int, optional
        Number of split-half resamples to assess during permutation testing.
        Default: 0\
    """),
    cross_val=dedent("""\
    test_split : int, optional
        Number of splits for generating test sets during cross-validation.
        Default: 100
    test_size : [0, 1) float, optional
        Proportion of data to partition to test set during cross-validation.
        Default: 0.25\
    """),
    covariance=dedent("""\
    covariance : bool, optional
        Whether to use the cross-covariance matrix instead of the cross-
        correlation during the decomposition. Only set if you are sure this is
        what you want as many of the results may become more difficult to
        interpret (i.e., :py:attr:`~.structures.PLSResults.behavcorr` will no
        longer be intepretable as Pearson correlation values). Default: False\
    """),
    rotate=dedent("""\
    rotate : bool, optional
        Whether to perform Procrustes rotations during permutation testing. Can
        inflate false-positive rates; see Kovacevic et al., (2013) for more
        information. Default: True\
    """),
    ci=dedent("""\
    ci : [0, 100] float, optional
        Confidence interval to use for assessing bootstrap results. This
        roughly corresponds to an alpha rate; e.g., the 95%ile CI is
        approximately equivalent to a two-tailed p <= 0.05. Default: 95\
    """),
    proc_options=dedent("""\
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed to use for random number generation. Helps ensure reproducibility
        of results. Default: None
    verbose : bool, optional
        Whether to show progress bars as the analysis runs. Note that progress
        bars will not persist after the analysis is completed. Default: True
    n_proc : int, optional
        How many processes to use for parallelizing permutation testing and
        bootstrap resampling. If not specified will default to serialized
        processing (i.e., one processor). Can optionally specify 'max' to use
        all available processors. Default: None\
    """),
    pls_results=dedent("""\
    results : :obj:`pyls.structures.PLSResults`
        Dictionary-like object containing results from the PLS analysis\
    """),
    resamples=dedent("""\
    permsamples : array_like, optional
        Re-sampling array to be used during permutation test (if n_perm > 0).
        If not specified a set of unique permutations will be generated.
        Default: None
    bootsamples : array_like, optional
        Resampling array to be used during bootstrap resampling (if n_boot >
        0). If not specified a set of unique bootstraps will be generated.
        Default: None\
    """),
    references=dedent("""\
    McIntosh, A. R., Bookstein, F. L., Haxby, J. V., & Grady, C. L. (1996).
    Spatial pattern analysis of functional brain images using partial least
    squares. NeuroImage, 3(3), 143-157.

    McIntosh, A. R., & Lobaugh, N. J. (2004). Partial least squares analysis of
    neuroimaging data: applications and advances. NeuroImage, 23, S250-S263.

    Krishnan, A., Williams, L. J., McIntosh, A. R., & Abdi, H. (2011). Partial
    Least Squares (PLS) methods for neuroimaging: a tutorial and review.
    NeuroImage, 56(2), 455-475.

    Kovacevic, N., Abdi, H., Beaton, D., & McIntosh, A. R. (2013). Revisiting
    PLS resampling: comparing significance versus reliability across range of
    simulations. In New Perspectives in Partial Least Squares and Related
    Methods (pp. 159-170). Springer, New York, NY. Chicago\
    """)
)


class PLSInputs(ResDict):
    allowed = [
        'X', 'Y', 'groups', 'n_cond', 'n_perm', 'n_boot', 'n_split',
        'test_split', 'test_size', 'mean_centering', 'covariance', 'rotate',
        'ci', 'seed', 'verbose', 'n_proc', 'bootsamples', 'permsamples',
        'method', 'n_components', 'aggfunc'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.get('n_split') == 0:
            self['n_split'] = None

        if self.get('test_split') == 0:
            self['test_split'] = None

        if self.get('n_proc') is not None:
            n_proc = self.get('n_proc')
            if n_proc == 'max' or n_proc == -1:
                self['n_proc'] = cpu_count()
            elif n_proc < 0:
                self['n_proc'] = cpu_count() + 1 + n_proc

        ts = self.get('test_size')
        if ts is not None and (ts < 0 or ts >= 1):
            raise ValueError('test_size must be in [0, 1). Provided value: {}'
                             .format(ts))


PLSInputs.__doc__ = dedent("""\
    PLS input information

    Attributes
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features.
    Y : (S, T) array_like
        Behavioral matrix, where `S` is observations and `T` is features.
        If from :obj:`.behavioral_pls`, this is the provided behavior matrix;
        if from :obj:`.meancentered_pls`, this is a dummy-coded group/condition
        matrix.
    {groups}
    {conditions}
    {mean_centering}
    {covariance}
    {stat_test}
    {rotate}
    {ci}
    {proc_options}
    """).format(**_pls_input_docs)


class PLSResults(ResDict):
    r"""
    Dictionary-like object containing results of PLS analysis

    Attributes
    ----------
    x_weights : (B, L) `numpy.ndarray`
        Weights of `B` features used to project `X` matrix into PLS-derived
        component space
    y_weights : (J, L) `numpy.ndarray`
        Weights of `J` features used to project `Y` matrix into PLS-derived
        component space; not available with :func:`.pls_regression`
    x_scores : (S, L) `numpy.ndarray`
        Projection of `X` matrix into PLS-derived component space
    y_scores : (S, L) `numpy.ndarray`
        Projection of `Y` matrix into PLS-derived component space
    y_loadings : (J, L) `numpy.ndarray`
        Covariance of features in `Y` with projected `x_scores`
    singvals : (L, L) `numpy.ndarray`
        Singular values for PLS-derived component space; not available with
        :func:`.pls_regression`
    varexp : (L,) `numpy.ndarray`
        Variance explained in each of the PLS-derived components
    permres : :obj:`~.structures.PLSPermResults`
        Results of permutation testing, as applicable
    bootres : :obj:`~.structures.PLSBootResults`
        Results of bootstrap resampling, as applicable
    splitres : :obj:`~.structures.PLSSplitHalfResults`
        Results of split-half resampling, as applicable
    cvres : :obj:`~.structures.PLSCrossValidationResults`
        Results of cross-validation testing, as applicable
    inputs : :obj:`~.structures.PLSInputs`
        Inputs provided to original PLS
    """
    allowed = [
        'x_weights', 'y_weights', 'x_scores', 'y_scores',
        'y_loadings', 'singvals', 'varexp',
        'permres', 'bootres', 'splitres', 'cvres', 'inputs'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create all sub-dictionaries
        self.inputs = PLSInputs(**kwargs.get('inputs', kwargs))
        self.bootres = PLSBootResults(**kwargs.get('bootres', kwargs))
        self.permres = PLSPermResults(**kwargs.get('permres', kwargs))
        self.splitres = PLSSplitHalfResults(**kwargs.get('splitres', kwargs))
        self.cvres = PLSCrossValidationResults(**kwargs.get('cvres', kwargs))


class PLSBootResults(ResDict):
    """
    Dictionary-like object containing results of PLS bootstrap resampling

    Attributes
    ----------
    x_weights_normed : (B, L) `numpy.ndarray`
        `x_weights` normalized by their standard error, obtained from bootstrap
        resampling (see `x_weights_stderr`)
    x_weights_stderr : (B, L) `numpy.ndarray`
        Standard error of `x_weights`, used to generate `x_weights_normed`
    y_loadings : (J, L) `numpy.ndarray`
        Covariance of features in `Y` with projected `x_scores`; not available
        with :func:`.meancentered_pls`
    y_loadings_boot : (J, L, R) `numpy.ndarray`
        Distribution of `y_loadings` across all bootstrap resamples; not
        available with :func:`.meancentered_pls`
    y_loadings_ci: (J, L, 2) `numpy.ndarray`
        Lower (..., 0) and upper (..., 1) bounds of confidence interval for
        `y_loadings`; not available with :func:`.meancentered_pls`
    contrast : (J, L) `numpy.ndarray`
        Group x condition averages of :attr:`brainscores_demeaned`. Can be
        treated as a contrast indicating group x condition differences. Only
        obtained from :obj:`.meancentered_pls`.
    contrast_boot : (J, L, R) `numpy.ndarray`
        Bootstrapped distribution of `contrast`; only available with
        :func:`.meancentered_pls`
    contrast_ci : (J, L, 2) `numpy.ndarray`
        Lower (..., 0) and upper (..., 1) bounds of confidence interval for
        `contrast`; only available with :func:`.meancentered_pls`
    bootsamples : (S, R) `numpy.ndarray`
        Indices of bootstrapped samples `S` across `R` resamples.
    """
    allowed = [
        'x_weights_normed', 'x_weights_stderr', 'bootsamples',
        'y_loadings', 'y_loadings_boot', 'y_loadings_ci',
        'contrast', 'contrast_boot', 'contrast_ci'
    ]


class PLSPermResults(ResDict):
    """
    Dictionary-like object containing results of PLS permutation testing

    Attributes
    ----------
    pvals : (L,) `numpy.ndarray`
        Non-parametric p-values used to examine whether components from
        original decomposition explain more variance than permuted components
    permsamples : (S, P) `numpy.ndarray`
        Resampling array used to permute `S` samples over `P` permutations
    """
    allowed = [
        'pvals', 'permsamples', 'perm_singval'
    ]


class PLSSplitHalfResults(ResDict):
    """
    Dictionary-like object containing results of PLS split-half resampling

    Attributes
    ----------
    ucorr, vcorr : (L,) `numpy.ndarray`
        Average correlations between split-half resamples in original (non-
        permuted) data for left/right singular vectors. Can be interpreted
        as reliability of `L` latent variables
    ucorr_pvals, vcorr_pvals : (L,) `numpy.ndarray`
        Number of permutations where correlation between split-half
        resamples exceeded original correlations, normalized by the total
        number of permutations. Can be interpreted as the statistical
        significance of the reliability of `L` latent variables
    ucorr_uplim, vcorr_uplim : (L,) `numpy.ndarray`
        Upper bound of confidence interval for correlations between split
        halves for left/right singular vectors
    ucorr_lolim, vcorr_lolim : (L,) `numpy.ndarray`
        Lower bound of confidence interval for correlations between split
        halves for left/right singular vectors
    """
    allowed = [
        'ucorr', 'vcorr',
        'ucorr_pvals', 'vcorr_pvals',
        'ucorr_uplim', 'vcorr_uplim',
        'ucorr_lolim', 'vcorr_lolim'
    ]


class PLSCrossValidationResults(ResDict):
    """
    Dictionary-like object containing results of PLS cross-validation testing

    Attributes
    ----------
    r_squared : (T, I) `numpy.ndarray`
        R-squared ("determination coefficient") for each of `T` predicted
        behavioral scores against true behavioral scores across `I` train /
        test split
    pearson_r : (T, I) `numpy.ndarray`
        Pearson's correlation for each of `T` predicted behavioral scores
        against true behavioral scores across `I` train / test split
    """
    allowed = [
        'pearson_r', 'r_squared'
    ]
