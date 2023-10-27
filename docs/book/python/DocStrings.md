(Chap_DocStrings)=
# Docstrings and Documentation

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

```{prf:observation} Eagleson's Law of Programming
:label: ObsDocStrings_Eagleson
> "Any code of your own that you haven't looked at for six or more months might as well have been written by someone else."[^EaglesonsLaw]
```

```{prf:observation} Guido van Rossum on clear code
:label: ObsDocStrings_Guido
> "Code is more often read than written."[^Guido]
```

Good documentation is critical to the ability of yourself and others to understand and disseminate your work and to allow others to reproduce it. As Eagleson's Law of Programming implies in {prf:ref}`ObsDocStrings_Eagleson` above, one of the biggest benefits of good documentation might be to the core maintainers and original code writers of a project. Despite the aspiration that the Python programming language be easy and intuitive enough to be its own documentation, we have often found than any not-well-documented code written by ourselves that is only a few months old is more likely to require a full rewrite rather than incremental additions and improvements.

Python scripts allow for two types of comments: inline comments (which are usually a line or two at a time) and docstrings, which are longer blocks set aside to document the source code. We further explore other more extensive types of documentation including README files, Jupyter notebooks, cloud notebooks, Jupyter Books, and published documentation forms.

A good resource is the RealPython tutorial entitled, "[Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)" {cite}`Mertz:2023`.


(SecDoc_comments)=
## Inline comments

"[PEP 257--Docstring Conventions](https://peps.python.org/pep-0257/)" differentiates between inline comments, which use the `#` symbol, and one-line docstrings, which use the `"""..."""` format {cite}`GoodgerVanRossum:2001`.  An block of code with inline comments might look like:

```python
# imports
import numpy as np

# create an array of zeros
zeros_array = np.zeros(10)
```

These types of comments are short and help to clarify what is to appear in the next few lines.  They help to remind others (or the future you) why you did something.  In this time of large language models, and [GitHub Copilot](https://github.com/features/copilot) they also provide valuable input for feed-forwrd models and make it much more likely the AI predicts your next line of code and writes it for you.

(SecDoc_docstrings)=
## Docstrings

Docstrings are longer blocks of comments that are set aside to document the source code.  Docstrings are usually multi-line and are enclosed in triple quotes `"""..."""`.  Docstrings are most often used at the top of a module to document what it does and the functions it containts and just after function or class definitions to document what they do.  Docstrings can also be used to document variables and other objects.  Docstrings can be accessed by the `help()` function and are used by the `pydoc` module to automatically generate documentation for your code.

The following is an example of a docstring for a function:

```python
def FOC_savings(c, r, beta, sigma):
    r"""
    Computes Euler errors for the first order condition for savings from
    the household's problem.

    .. math::
        c_{t}^{-\sigma} = \beta (1 + r_{t+1}) c_{t+1}^{-\sigma}

    Args:
        c (array_like): consumption in each period
        r (array_like): the real interest rate in each period
        beta (scalar): discount factor
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        euler (Numpy array): Euler error from FOC for savings

    """
    if sigma == 1:
        muc = 1 / c
    else:
        mu_c = c ** (-sigma)
    euler_error = mu_c[:-1] - beta * (1 + r[1:]) * mu_c[1:]

    return euler_error
```

A few notes on this documentation of the `FOC_savings` function.  First, see that the docstring starts of with  a clear description of what the function does.  Second, you can see the `:math` tags that allow you to write [LaTeX](https://www.latex-project.org) equations that will be rendered in the documentation. Docstrings written using [reStructuredText](https://docutils.sourceforge.io/rst.html) markup can be compiled through various packages to render equations and other formatting options. Third, the `Args` and `Returns` sections are used to document the arguments and return values of the function.

"[PEP 257--Docstring Conventions](https://peps.python.org/pep-0257/)" give suggested format and usage for docstrings in Python {cite}`GoodgerVanRossum:2001`.  And there are two main styles for writing docstrings, the [Google style]*(https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and the [NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).  While there are other ways to write docstrings (even those that meet PEP 257 standards), these two styles are so commonly used and are compatible with the Sphinx documentation package that we recommend using one of these two styles.  `OG-Core` used the Google style, so you might adopt that to be consistent.


(SecDoc_README)=
## README file

`README` files are a common way to document software.  These are typically plain text files that include file structures and instructions on running the software.

If your project is hosted on GitHub, it would make sense to write the `README` file in Markdown.  Markdown is a lightweight markup language that is easy to read and write and can be rendered in HTML, a very nice feature when you have this file on the internet via GitHub.  Markdown is used in many places including GitHub, Jupyter notebooks, and Jupyter Book documentation.  See the [Markdown Guide](https://www.markdownguide.org) for more information on Markdown.  And you can see an example of a `README` file written with Markdown in the `OG-Core` repository [here](https://github.com/PSLmodels/OG-Core/#readme).


(SecDoc_JupNote)=
## Jupyter notebooks

As discussed in the {ref}`Chap_PythonIntro` Chapter, Jupyter notebooks are a great way to interactively work with Python.  They are also a great way to document your work.  You can write Markdown in cells of these notebooks to provide text around your blocks of code. This Markdown can then be compiled to render nice formatting. You can also use the code cells to document your code just as you would in any Python script.  You can see an example of a Jupyter notebook in the Cost-of-Capital-Calculator` repository [here](https://github.com/PSLmodels/Cost-of-Capital-Calculator/blob/master/docs/book/content/examples/PSL_demo.ipynb).  As you can see in that example, Jupyter notebooks are rendered as HTML on GitHub, making viewing them easy.


(SecDoc_CloudNote)=
## Cloud notebooks

[Google Colab](https://colab.research.google.com) provides cloud-hosting for Jupyter notebooks.  These have all the same functionality as locally hosted notebooks described above, but they are hosted on Google's servers.  This allows you to share your work with others and to collaborate on projects.  It also means you can run Python (or other languages) without the need to install any special software on your machine.  You just need a browser, internet connection, and Google account.


(SecDoc_JupBook)=
## Jupyter Book documentation

For long and detailed documentation, [Jupyter Books](https://jupyterbook.org/en/stable/intro.html) are a great option.  Jupyter Books are a collection Markdown, ReStructuredText, [MyST](https://mystmd.org) files and Jupyter notebooks that are compiled into a book format.  Jupyter Books can be compiled to HTML or PDF formats, making them easy to share.  This training guide was created in Jupyter Book!

[TODO: Show the slick rst interface between Sphinx and the OG-Core modules that automatically compile LaTeX documentation into the Jupyter Book API documentation. See this Jupyter Book API chapter on [Firms](https://pslmodels.github.io/OG-Core/content/api/firm.html) and the code that created it in [this folder](https://github.com/PSLmodels/OG-Core/tree/master/docs/book/content/api).]


(SecDoc_Other)=
## Other published documentaiton

Put discusion of other forms of published documentation here such as white papers, peer-reviewed articles, websites (readthedocs by Sphinx).


(SecDocstringExercises)=
## Exercises

```{exercise-start}
:label: ExerDoc-google
:class: green
```
Take a function your wrote in your solution to {numref}`ExerScipy-BM72_ss`.  Add a docstring to this function that uses the Google style.
```{exercise-end}
```


(SecDocstringFootnotes)=
## Footnotes

The footnotes from this chapter.

[^EaglesonsLaw]: We could not find a proper citation for the source of this quote "Eagleson's Law of Programming". Some entries on this thread entitled "[Who is Eagleson and where did Eagleson's law originate?](https://ask.metafilter.com/200910/Who-is-Eagleson-and-where-did-Eaglesons-law-originate)" suggest that the quote is at least as old as 1987 and is likely from [Peter S. Eagleson](https://en.wikipedia.org/wiki/Peter_S._Eagleson), a member of the MIT faculty since 1952. However, neither the date, nor the author is confirmed.

[^Guido]: This is a quote from Guido van Rossum, the original creator of the Python programming language, supposedly from an early PyCon conference. This quote is referenced in one of the early Python Enhancement Proposals, "[PEP 8--Style Guide for Python Code](https://peps.python.org/pep-0008/)" {cite}`VanRossumEtAl:2001`.
