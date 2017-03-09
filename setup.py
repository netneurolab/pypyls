from setuptools import setup, find_packages

__version__ = '0.0.01'

setup(
    name="pyls",
    version=__version__,
    description="A pythonic PLSC toolbox",
    maintainer="Ross Markello",
    maintainer_email="rossmarkello@gmail.com",
    url="http://github.com/rmarkello/pyls",
    install_requires=['numpy','scipy','scikit-learn'],
    packages=find_packages(exclude=['pyls/tests']),
    package_data={'pyls' : ['data/*'],
                  'pyls.tests' : ['data/*']},
    tests_require=['pytest'],
    download_url="https://github.com/rmarkello/pyls/archive/{0}.tar.gz".format(__version__),
    license='MIT')
