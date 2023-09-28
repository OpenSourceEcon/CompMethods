"""This file contains the fiscalsim-us package's metadata and dependencies."""

from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="UN-OG-Training",
    version="0.0.0",
    author="Jason DeBacker and Richard W. Evans",
    author_email="rickecon@gmail.com",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="UN open source OG-Core overlapping generations macroeconomic model training",
    keywords="tax benefit macroeconomic dynamic general equilibrium fiscal",
    license="http://www.fsf.org/licensing/licenses/agpl-3.0.html",
    url="https://github.com/OpenRG/UN-OG-Training",
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
        "ogcore",
        "linecheck",
        "yaml-changelog",
    ],
    # Windows CI requires Python 3.9.
    python_requires=">=3.10",
    tests_require=["pytest"],
    packages=find_packages(),
)
