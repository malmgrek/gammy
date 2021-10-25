import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "README.md")) as fh:
    long_description = fh.read()


setup(
    name="gammy",
    version="0.4.3",
    author="Stratos Staboulis",
    description="Generalized additive models with a Bayesian twist",
    url="https://github.com/malmgrek/gammy",
    packages=find_packages(exclude=["contrib", "doc", "tests"]),
    install_requires=[
        "numpy>=1.10.0",
        "scipy>=0.13.0",
        "bayespy",
        "h5py",
    ],
    extras_require={
        "test": ["pytest"],
    },
    keywords="bayesian statistics modeling gaussian process splines",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # MIT License
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
