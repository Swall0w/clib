#!usr/bin/env python

from setuptools import setup, find_packages

setup(name="clib",
    version="0.15",
    description="chainer module",
    url="https://github.com/Swall0w/clib",
    packages=find_packages(),
    entry_points="""
    [console_scripts]
    pig = pig.pig:main
    """,
    )
