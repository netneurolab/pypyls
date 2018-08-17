__version__ = '0.0.1'

NAME = 'pyls'
MAINTAINER = 'Ross Markello'
EMAIL = 'rossmarkello@gmail.com'
VERSION = __version__
LICENSE = 'MIT'
DESCRIPTION = ('A toolbox for performing multivariate decomposition analyses')
LONG_DESCRIPTION = ('')
URL = 'http://github.com/rmarkello/pyls'
DOWNLOAD_URL = ('https://github.com/rmarkello/{name}/archive/{ver}.tar.gz'
                .format(name=NAME, ver=__version__))

INSTALL_REQUIRES = [
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
    'plotting': ['pandas', 'seaborn']
}

PACKAGE_DATA = {
    'pyls': ['tests/data']
}
