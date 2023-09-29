(Chap_PythonIntro)=
# Introduction to Python

The `OG-Core` model, as well as the country-specific calibration packages, are written in the Python programming language. These training materials will provide you with sufficient background with Python and some of its most-used packages for data science that you will be able to understand and contribute to the source code underlying `OG-Core` and the calibration packages. We assume some basic background with mathematics, economics, and programming, but we do not assume any prior knowledge of Python.

As we walk you through the basics of Python, we will leverage some excellent open source materials put together by [QuantEcon](https://quantecon.org/) and the [Applied and Computational Mathematics Emphasis at BYU (BYU ACME)](https://acme.byu.edu/). And while we will point you to their tutorials, we have customized all our excercises to be relevant to the `OG-Core` model and the calibration packages.


(SecPythonIntroOverview)=
## Overview of Python
The Python.org site has documentation essays, one of which is entitled "[What is Python? Executive Summary](https://www.python.org/doc/essays/blurb/)". The first paragraph contains the following description.

> Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.

In addition to the description above, Python is an open source programming language that is freely available and customizable (see https://www.python.org/downloads/source/).

Python has some built in functionality with the standard library, but most of the functionality comes from packages that are developed by the open source community. The most important packages for data science are: NumPy, SciPy, Pandas, and Matplotlib.  We will introduce each of these packages as we go through the training materials as they are used heavily in `OG-Core` and the corresponding country calibration packages.


(SecPythonIntroInstall)=
## Installing Python
We recommend that you download the Anaconda distribution of Python provided by [Anaconda](https://www.anaconda.com/download). We also recommend the most recent stable version of Python, which is currently Python 3.11. This can be done from the [Anaconda download page](https://www.anaconda.com/download) for Windows, Mac OSX, and Linux machines. The code we will be writing uses common Python libraries such as `NumPy`, `SciPy`, `pickle`, `os`, `matplotlib`, and `time`, which are all included in the Anaconda distribution. If you are using a different distribution of Python, you may need to install these packages separately.


(SecPythonIntroWorkingWith)=
## Working with Python

There are several ways to interact with Python:
1. [Jupyter Notebook](https://jupyter.org/)
2. iPython session
3. Running a Python script from the command line
4. Running a Python script from an IDE such as [Spyder](https://www.spyder-ide.org/).

In our recommended Python development workflow, you will write Python scripts and modules (`*.py` files) in a text editor. Then you will run those scripts from your terminal. You will want a capable text editor for developing your code. Many capable text editors exist, but we recommend [Visual Studio Code](https://code.visualstudio.com) (VS Code). As you learn Python and complete the exercises in this training program, you will also use Python interactively in a Jupyter Notebook or iPython session. VS Code will be helpful here as well as it will allow you open Jupyter Notebooks and run Python interactively through the text editor.

VS Code is free and will be included with your installation of Anaconda. This is a very capable text editor and will include syntax highlighting for Python and and built in Git controls. In addition to the basics, you may want to use a more advanced linter for Python. This will help you correct syntax errors on the fly and provide helpful information as you declare objects and call functions. [This link](https://code.visualstudio.com/docs/python/linting) provides step-by-step instructions on using more advanced linting in VS Code.

Some extensions that we recommend installing into your VS Code:
* cornflakes-linter
* Git Extension Pack
* GitLens
* Jupyter
* Markdown All in One
* Pylance

In addition, [GitHub Copilot](https://github.com/features/copilot) is an amazing resource and can be added as an extension to VS Code. However, this service is not free of charge and does require an internet connection to work.

```{exercise-start}
:label: ExerPythonIntro
```
Read the BYU ACME "[Introduction to Python](https://acme.byu.edu/00000181-448a-d778-a18f-dfcae22f0001/intro-to-python)" lab and complete Problems 1 through 8 in the lab. {cite}`BYUACME_PythonIntro`
```{exercise-end}
```

[Put a review of Python data types here: string, byte, float, list, set, dict. Discuss mutability versus immutability as well as iterability. Also introduce NumPy arrays and Pandas DataFrames.]


(SecPythonIntroPackages)=
## Python Packages

When using `OG-Core` there are a handful of Python packages that will be useful and that these training materials will cover:

1. The Standard Library
2. NumPy for numerical computing (e.g., arrays, linear algebra, etc.)
3. Pandas for data analysis
4. Matplotlib for plotting
5. SciPy for scientific computing (e.g., optimization, interpolation, etc.)

All of these will be included as part of your installation of Anaconda.  Anaconda also includes a package manager called `conda` that will allow you to install additional packages and well help keep versions of packages consistent with each other.  We will not cover this in these training materials, but you can find more information about `conda` [here](https://docs.conda.io/en/latest/) and you'll find references to `conda` as we install the `OG-Core` package in the latter part of these training materials.


(SecPythonIntroTopics)=
## Python Training Topics

1. [Python Standard Library](StandardLibrary.md)
2. [Object Oriented Programming](OOP.md)
3. [NumPy](NumPy.md)
4. [Pandas](Pandas.md)
5. [Matplotlib](Matplotlib.md)
6. [SciPy](SciPy.md)
7. [Doc strings and comments](DocStrings.md)
8. [Unit testing in Python](UnitTesting.md)


<!-- (SecPythonIntroFootnotes)=
## Footnotes -->

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
