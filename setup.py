import os
from setuptools import setup, find_packages
import versioneer


if __name__ == "__main__":

    def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

    meta = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'gammy', '_meta.py')) as fp:
        exec(fp.read(), meta)


    setup(
        name         = "gammy",
        version      = versioneer.get_version(),
        author       = meta["__author__"],
        author_email = meta["__contact__"],
        description  = "Generalized additive models with a Bayesian twist",
        url          = "https://github.com/malmgrek/Gammy",
        cmdclass     = versioneer.get_cmdclass(),
        packages     = find_packages(),
        install_requires = [
            "bayespy",
            "h5py",
            "matplotlib",
            "numpy",
            "scipy"
        ],
        extras_require = {
            "dev": [
                "versioneer",
                "pytest",
                "hypothesis",
            ],
        },
        keywords     = [
            "Statistical modeling",
            "Bayesian statistics",
            "Machine learning",
        ],
        classifiers = [
            "Programming Language :: Python :: 3 :: Only",
            "Development Status :: 1 - Planning",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: {0}".format(meta["__license__"]),
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
        ],
        long_description = read('README.md'),
        long_description_content_type = "text/markdown",
    )
