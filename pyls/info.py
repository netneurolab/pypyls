# -*- coding: utf-8 -*-

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'pyls developers'
__copyright__ = 'Copyright 2018, pyls developers'
__credits__ = ['Elizabeth DuPre', 'Ross Markello']
__license__ = 'GPLv2'
__maintainer__ = 'Ross Markello'
__email__ = 'rossmarkello@gmail.com'
__status__ = 'Prototype'
__url__ = 'http://github.com/rmarkello/pyls'
__packagename__ = 'pyls'
__description__ = ('pyls is a Python toolbox for performing multivariate '
                   'decomposition analyses.')
__longdesc__ = 'README.md'
__longdesctype__ = 'text/markdown'

DOWNLOAD_URL = (
    'https://github.com/rmarkello/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

REQUIRES = [
    'numpy',
    'scikit-learn',
    'scipy',
    'tqdm'
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

EXTRAS_REQUIRE = {
    'plotting': [
        'pandas',
        'seaborn'
    ],
    'doc': [
        'sphinx>=1.2',
        'sphinx_rtd_theme'
    ],
    'tests': TESTS_REQUIRE,
}

EXTRAS_REQUIRE['all'] = list(
    set([v for deps in EXTRAS_REQUIRE.values() for v in deps])
)

PACKAGE_DATA = {
    'pyls': [
        'tests/data/*'
    ]
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]
