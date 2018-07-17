#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='pytorch-cnn-visualization',
    version='0.0',
    description='pytorch implementation of CNN visualization techniques',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.14.5',
        'opencv-python==3.4.1.15'
        'torch==0.4.0',
        'torchvision==0.2.1',
    ],
    extras_require={
        'dev': [
            'matplotlib',
            'ipdb',
            'flake8',
            'pylint',
            'pep8',
            'mypy',
            'pytest',
            'pytest-asyncio'
        ],
        'test': [
            'pytest',
            'pytest-asyncio'
        ],
    },
)
