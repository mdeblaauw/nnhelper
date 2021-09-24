from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'Neural network helper'
LONG_DESCRIPTION = 'Neural network helper for Pytorch'

setup(
    name='nnhelper',
    version=VERSION,
    author='Mark de Blaauw',
    author_email='mdeblaauw@mobiquity.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(include=['nnhelper', 'nnhelper.*']),
    install_requires=[
        'torch>=1.9.1',
        'numpy>=1.21.2'
    ],
    keywords=['python', 'neural network', 'pytorch']
)