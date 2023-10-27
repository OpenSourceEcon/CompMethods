(Chap_PythonIntro)=
# Introduction to Python

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

Many models are written in the Python programming language. Python is the 2nd most widely used language on all GitHub repository projects {cite}`GitHub:2022`, and Python is the 1st most used programming language according to the PYPL ranking of September 2023 {cite}`Stackscale:2023`.

As these tutorials walk you through the basics of Python, they will leverage some excellent open source materials put together by [QuantEcon](https://quantecon.org/) and the [Applied and Computational Mathematics Emphasis at BYU (BYU ACME)](https://acme.byu.edu/2023-2024-materials). And while the tutorials will point you to those of these other organizations, we have customized all our excercises to be relevant to the work and research of economists.


(SecPythonIntroOverview)=
## Overview of Python
The Python.org site has documentation essays, one of which is entitled "[What is Python? Executive Summary](https://www.python.org/doc/essays/blurb/)". The first paragraph contains the following description.

> Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.

In addition to the description above, Python is an open source programming language that is freely available and customizable (see https://www.python.org/downloads/source/).

Python has some built in functionality with the standard library, but most of the functionality comes from packages that are developed by the open source community. The most important packages for data science are: NumPy, SciPy, Pandas, and Matplotlib.  We will introduce each of these packages as we go through the training materials as they are used heavily in economics applications.


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

In the iframe below is a PDF of the BYU ACME open-access lab entitled, "Python Intro". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_PythonIntro`. {numref}`ExerPythonIntro` below has you work through the problems in this BYU ACME lab. The Python code file ([`python_intro.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/PythonIntro/python_intro.py)) used in the lab is stored in the [`./code/PythonIntro/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/PythonIntro) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1CHl8C-QKgs8jHzsRfJSMWkVqq0elzP1F/preview?usp=sharing">
  </iframe>
</div>

We cover Python's built-in functions, constants, and data types and their properties in {numref}`ExerStandardLibrary` of the {ref}`Chap_StdLib` chapter. We also introduce different commonly used objects like Numpy arrays and operations in chapter {ref}`Chap_Numpy` and Pandas DataFrames and operations in chapter {ref}`Chap_Pandas`.


(SecPythonIntroPackages)=
## Python Packages

Economics applications heavily use a handful of Python packages that will be useful and that these training materials will cover:

1. The Standard Library
2. NumPy for numerical computing (e.g., arrays, linear algebra, etc.)
3. Pandas for data analysis
4. Matplotlib for plotting
5. SciPy for scientific computing (e.g., optimization, interpolation, etc.)

All of these will be included as part of your installation of Anaconda. Anaconda also includes a package manager called `conda` that will allow you to install additional packages and well help keep versions of packages consistent with each other.  We will not cover this in these training materials, but you can find more information about `conda` [here](https://docs.conda.io/en/latest/) and you'll find references to `conda` as we install packages throughout these training materials.


(SecPythonIntroTopics)=
## Python Training Topics

1. [Python Standard Library](StandardLibrary.md)
2. [Exception handling and file input/output](ExceptionsIO.md)
3. [Object Oriented Programming](OOP.md)
4. [NumPy](NumPy.md)
5. [Pandas](Pandas.md)
6. [Matplotlib](Matplotlib.md)
7. [SciPy](SciPy.md)
8. [Doc strings and documentation](DocStrings.md)
9. [Unit testing](UnitTesting.md)


(SecPythonIntroUnix)=
## (Optional): Using the Unix Shell

Unix is an old operating system that is the basis for the Linux and Mac operating systems. Many Python users with Mac or Linux operating systems follow a workflow that includes working in the terminal and using Unix commands. This section is optional because Windows terminals do not have the same Unix commands. For those interested, feel free to work through the Unix lab below from BYU ACME. This lab features great examples and instruction, and also has seven good exercises for you to practice on.

In the iframe below is a PDF of the BYU ACME open-access lab entitled, "Unix Shell 1: Introduction". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Unix1`. {numref}`ExerUnix1` below has you work through the problems in this BYU ACME lab. The shell script file ([`unixshell1.sh`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnixShell1/unixshell1.sh)) used in the lab, along with the associated zip file ([`Shell1.zip`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnixShell1/Shell1.zip)), are stored in the [`./code/UnixShell1/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnixShell1) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/18eTLp_FhWFYgAItIZnX6gesIvg91rXW5/preview?usp=sharing">
  </iframe>
</div>

(SecPythonIntroExercises)=
## Exercises

```{exercise-start} Python introduction
:label: ExerPythonIntro
:class: green
```
Read the BYU ACME "[Introduction to Python](https://drive.google.com/file/d/1CHl8C-QKgs8jHzsRfJSMWkVqq0elzP1F/view?usp=sharing)" lab and complete Problems 1 through 8 in the lab. {cite}`BYUACME_PythonIntro`
```{exercise-end}
```

```{exercise-start} OPTIONAL: Unix shell commands
:label: ExerUnix1
:class: green
```
Read the BYU ACME "[Unix Shell 1: Introduction](https://drive.google.com/file/d/18eTLp_FhWFYgAItIZnX6gesIvg91rXW5/view?usp=sharing)" lab and complete Problems 1 through 7 in the lab. {cite}`BYUACME_Unix1`
```{exercise-end}
```
