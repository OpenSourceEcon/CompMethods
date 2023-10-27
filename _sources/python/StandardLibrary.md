(Chap_StdLib)=
# Python Standard Library

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

The **standard library** of Python is all the built-in functions of the programming language as well as the modules included with the most common Python distributions. The Python online documentation has an [excellent page](https://docs.python.org/3/library/index.html) describing the standard library. These functionalities include built-in [functions](https://docs.python.org/3/library/functions.html), [constants](https://docs.python.org/3/library/constants.html), and [object types](https://docs.python.org/3/library/stdtypes.html), and [data types](https://docs.python.org/3/library/datatypes.html). We recommend that you read these sections in the Python documentation.

In addition, the iframe below contains a PDF of the BYU ACME open-access lab entitled, "The Standard Library". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_StandardLibrary`. {numref}`ExerStandardLibrary` below has you work through the problems in this BYU ACME lab. The two Python files used in this lab are stored in the [`./code/StandardLibrary/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/StandardLibrary) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1JT2TolhLhyQBO2iyGoBZYVPgni0dc3x6/preview?usp=sharing">
  </iframe>
</div>


(SecStdLibExercises)=
## Exercises

```{exercise-start}
:label: ExerStandardLibrary
:class: green
```
Read the BYU ACME "[The Standard Library](https://drive.google.com/file/d/1JT2TolhLhyQBO2iyGoBZYVPgni0dc3x6/view?usp=sharing)" lab and complete Problems 1 through 5 in the lab. {cite}`BYUACME_StandardLibrary`
```{exercise-end}
```

```{exercise-start}
:label: ExerStd-module_run
:class: green
```
Create a python module that prints something (e.g. `Hello World!`) and run it from the command line using `python module_name.py`.
```{exercise-end}
```

```{exercise-start}
:label: ExerStd-notebook_run
:class: green
```
Create a Jupyter notebook (`.ipynb`) with your Python code from {numref}`ExerStd-module_run` and run it in the VS Code text editor.
```{exercise-end}
```

```{exercise-start}
:label: ExerStd-def_function
:class: green
```
Write a function that finds the Fibonacci sequence up to an integer `N` > 0 in the notebook.  Now call this function for `N = 10` and `N=100`.
```{exercise-end}
```

```{exercise-start}
:label: ExerStd-sys
:class: green
```
Use the `sys` module to create a relative path from a Python module, print that path.
```{exercise-end}
```
