"""This file contains the CompMethods package's metadata and dependencies."""

from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="CompMethods",
    version="0.0.2",
    author="Richard W. Evans",
    author_email="rickecon@gmail.com",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="Computational Methods for Economists using Python",
    keywords="code python git github data science python regression causal inference structural estimation machine learning neural networks deep learning statistics econometrics",
    license="http://www.fsf.org/licensing/licenses/agpl-3.0.html",
    url="https://github.com/OpenSourceEcon/CompMethods",
    include_package_data=True,  # Will read MANIFEST.in
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "ipython",
        "matplotlib",
        "bokeh",
        "sphinx",
        "sphinx-argparse",
        "sphinx-exercise",
        "sphinxcontrib-bibtex>=2.0.0",
        "sphinx-math-dollar",
        "pydata-sphinx-theme",
        "jupyter-book>=0.11.3",
        "jupyter",
        "black",
        "setuptools",
        "pytest",
        "coverage",
        "linecheck",
        "yaml-changelog",
    ],
    python_requires=">=3.10",
    tests_require=["pytest"],
    packages=find_packages(),
)
