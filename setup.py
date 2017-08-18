#!usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="clib",
    version="0.16",
    description="Neural Network module with chainer",
    url="https://github.com/Swall0w/clib",
    install_requires=['numpy','chainer'],
    license=license,
    packages=find_packages(exclude=('tests')),
    test_suite='tests',
    entry_points="""
    [console_scripts]
    pig = pig.pig:main
    """,
    )
