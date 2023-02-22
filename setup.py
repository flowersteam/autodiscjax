#! /usr/bin/python3
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='autodiscjax',
    version='0.3',
    author='Mayalen Etcheverry',
    author_email='mayalen.etcheverry@inria.fr',
    description=' python software built upon jax, that allows to perform automated discovery and exploration of biological networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/flowersteam/autodiscjax',
    license='MIT',
    packages = ['autodiscjax'],
    install_requires=['jax[cpu]', 'optax', 'equinox', 'addict', 'experiment-utils', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

