import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.md")) as f:
    readme = f.read()


about = {}
with open(os.path.join(here, "gammy", "__version__.py"), "r") as f:
    exec(f.read(), about)


setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    description=about["__description__"],
    url=about["__url__"],
    packages=find_packages(exclude=["contrib", "doc", "tests"]),
    license=about["__license__"],
    install_requires=[
        "numpy>=1.10.0",
        "scipy>=0.13.0",
        "bayespy",
        "h5py",
    ],
    extras_require={
        "test": ["pytest"],
        "doc": ["sphinx", "numpydoc"]
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
    long_description=readme,
    long_description_content_type="text/markdown",
)
