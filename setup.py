#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    import versioneer
    from io import open
    import os.path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from pyls.info import (
        __author__,
        __description__,
        __email__,
        __license__,
        __longdesc__,
        __longdesctype__,
        __maintainer__,
        __packagename__,
        __url__,
        __version__,
        CLASSIFIERS,
        DOWNLOAD_URL,
        EXTRAS_REQUIRE,
        PACKAGE_DATA,
        REQUIRES,
        TESTS_REQUIRE,
    )

    root_dir = op.dirname(op.abspath(getfile(currentframe())))

    version = None
    cmdclass = {}
    if op.isfile(op.join(root_dir, 'pyls', 'VERSION')):
        with open(op.join(root_dir, 'pyls', 'VERSION')) as vfile:
            version = vfile.readline().strip()
        PACKAGE_DATA['pyls'].insert(0, 'VERSION')

    if version is None:
        version = versioneer.get_version()
    cmdclass = versioneer.get_cmdclass()

    # get long description from README
    with open(op.join(root_dir, __longdesc__)) as src:
        __longdesc__ = src.read()

    setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        long_description_content_type=__longdesctype__,
        author=__author__,
        author_email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=CLASSIFIERS,
        download_url=DOWNLOAD_URL,
        install_requires=REQUIRES,
        packages=find_packages(exclude=['pyls/tests']),
        package_data=PACKAGE_DATA,
        tests_require=TESTS_REQUIRE,
        extras_require=EXTRAS_REQUIRE,
        cmdclass=cmdclass
    )


if __name__ == '__main__':
    main()
