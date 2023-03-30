#! /usr/bin/python3
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='autodiscjax',
    version='0.4',
    author='Mayalen Etcheverry',
    author_email='mayalen.etcheverry@inria.fr',
    description=' python library built on top of jax to facilitate automated exploration and simulation of computational models of biological processes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/flowersteam/autodiscjax',
    license='MIT',
    packages=setuptools.find_packages(exclude=["examples", "test"]),
    install_requires=['jax[cpu]', 'optax', 'equinox', 'addict'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

