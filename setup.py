#!usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="clib",
    version="0.1.8",
    description="Neural Network module with chainer",
    url="https://github.com/Swall0w/clib",
    install_requires=['numpy', 'chainer', 'scikit-image'],
    license=license,
    packages=find_packages(exclude=('tests')),
    test_suite='tests',
    entry_points="""
    [console_scripts]
    pig = pig.pig:main
    """,
    )
