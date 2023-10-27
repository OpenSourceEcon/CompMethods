(Chap_Numpy)=
# NumPy

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

NumPy is Python's fundamantal numerical package (the name stands for "numerical Python"), and is at the basis of most computation using Python.[^NumPy] Our discussion of Python's NumPy package starts with Travis Oliphant, who was the primary creator of the NumPy package, a founding contributor to Python's SciPy package (covered in the {ref}`Chap_SciPy` chapter), founder of [Anaconda, Inc.](https://www.anaconda.com/) that maintains the most popular distribution of Python, and a co-founder of the [NumFOCUS](https://numfocus.org/) non-profit that fiscally supports some of the primary package projects in Python.[^Oliphant]

Oliphant was a mathematics and electrical engineering student who came up through his masters degree using MATLAB with a focus primarily on signal processing. While working on a PhD, he needed to create custom code that could do signal processing operations that had never been done before. These operations required combinations of mathematical operations. Oliphant liked the ideas of network and collaboration in the open source software community, and Python was a language that felt intuitive and comfortable to him. However, Python had no established numerical matrix operations libraries. Oliphant created the NumPy package to be that numerical engine based on linear algebra array operations.

The fundamental object of the NumPy package is the NumPy array [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html). Python's native objects---such as lists, tuples, and dictionaries---can hold numbers and perform operations on those numbers. But the NumPy array allows for storing high-dimensional arrays of numbers on which linear algebra and tensor functions can be operated. These linear algebra operations are more effecient than working with lists and tuples, and they form the foundation of modern optimization and machine learning. Learning to use Python's NumPy package is an essential skill for many numerical computations and other operations.

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "Introduction to NumPy". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_NumPy1`. {numref}`ExerNumPy-acme1` below has you work through the problems in this BYU ACME lab. A Python file template ([`numpy_intro.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/NumPyIntro/numpy_intro.py)) and a matrix data file ([`grid.npy`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/NumPyIntro/grid.npy)) used in the lab are stored in the [`./code/NumPyIntro/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/NumPyIntro) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1Hj3ok81gJAxcUTHh_8BrxX-B4belupPN/preview?usp=sharing">
  </iframe>
</div>

The following iframe contains a PDF of the BYU ACME open-access lab entitled, "Advanced NumPy", which contains content and exercises that build off of the previous BYU ACME NumPy lab. You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_NumPy2`. {numref}`ExerNumPy-acme2` below has you work through the problems in this BYU ACME lab. A Python file template ([`advanced_numpy.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/AdvancedNumPy/_advanced_numpy.py)) used in the lab are stored in the [`./code/AdvancedNumPy/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/AdvancedNumPy) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/15KxliSp0C_mLf7TrLQbnC0wO4YaK7ePi/preview?usp=sharing">
  </iframe>
</div>


(SecNumPyExtensions)=
## Extensions and future paths

One of the drawbacks to the degree to which NumPy arrays are fundamental to Python's numerical computing is that the format of those arrays is a requirement in Python's most highly used scientific computing and machine learning packages ({ref}`Chap_SciPy` and scikit-learn). However, advances in hardware, large data methods, and optimization algorithms now take much more advantage of parallel computing algorithms, hybrid architectures across multiple traditional processors and GPU's. All of these innovations have been difficult to incorporate into Python's scientific computing stack because NumPy arrays have been difficult to make flexible to these architectures.

Below are three areas that have been working to make Python better on these dimentions.
* Dask arrays
* QuantSight development and support of array API's in SciPy and in scikit-learn. See here for the [scikit-learn blog post](https://labs.quansight.org/blog/array-api-support-scikit-learn). And see here for the [SciPy blog post](https://labs.quansight.org/blog/scipy-array-api).
* Modular's development of the Mojo programming language.


(SecNumPyExercises)=
## Exercises

```{exercise-start}
:label: ExerNumPy-acme1
:class: green
```
Read the BYU ACME "[Introduction to NumPy](https://drive.google.com/file/d/1Hj3ok81gJAxcUTHh_8BrxX-B4belupPN/view?usp=sharing)" lab and complete Problems 1 through 7 in the lab. {cite}`BYUACME_NumPy1`
```{exercise-end}
```

```{exercise-start}
:label: ExerNumPy-acme2
:class: green
```
Read the BYU ACME "[Advanced NumPy](https://drive.google.com/file/d/15KxliSp0C_mLf7TrLQbnC0wO4YaK7ePi/view?usp=sharing)" lab and complete Problems 1 through 7 in the lab. {cite}`BYUACME_NumPy2`
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-array
:class: green
```
Create a Numpy array `b` (defined this as the savings of 2 agents (the rows) over 5 periods (the columns)):
\begin{equation*}
  b= \begin{bmatrix}
       1.1 & 2.2 & 3.0 & 2.0 & 1.0 \\
       3.3 & 4.4 & 5.0 & 3.7 & 2.0
     \end{bmatrix}
\end{equation*}
Use the `shape` method of NumPy arrays to print the shape of this matrix.  Use array slicing to print the first row of `b`, which represents the lifecycle savings decisions of the first agent (i.e., the amount they choose to save in each of their 5 periods of life). Use array slicing to print the second column of `b`, which is the savings of both agents when they are in their second period of life. Finally, use array slicing to print the first two rows and the last three columns of `b` (i.e., the savings of both agents from middle age onwards).
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-dotproduct
:class: green
```
Now let's think about the matrix `b` as representing not two individual agents, but two types of agents who each live for five periods. In this way, we will interpret the values in `b` as the total savings of different cohorts of these two types of agents who are all living together at a point in time. Now, define a matrix `Omega`:
\begin{equation*}
  \Omega=
    \begin{bmatrix}
      0.05 & 0.05 & 0.08 & 0.06 & 0.2 \\
      0.12 & 0.16 & 0.03 & 0.2 & 0.05
    \end{bmatrix}
\end{equation*}
`Omega` represents the fraction of agents in the economy of each type/cohort (Note that the elements of `Omega` sum to 1). Use matrix multiplication to find `B`, which is the dot product of `b` and the transpose of `Omega`.
\begin{equation*}
  B = b\Omega^T
\end{equation*}
Print your matrix `B`. What is its shape? What does `B` represent?
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-mult
:class: green
```
Multiply element-wise (Hadamard product) the matrix `b` from {numref}`ExerNumpy-array` by the matrix `Omega` from {numref}`ExerNumpy-dotproduct`. Use the `numpy.array.sum()` method on the resulting matrix, with the appropriate `axis` argument in the parentheses to find the total savings of each cohort.
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-zeros
:class: green
```
In one line, create a matrix of zeros that is the same size as `b` from {numref}`ExerNumpy-array`.
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-where
:class: green
```
Use `numpy.where` to return the elements of `b` from {numref}`ExerNumpy-array` that are greater than 2.0 and zero elsewhere.
```{exercise-end}
```

```{exercise-start}
:label: ExerNumpy-stack
:class: green
```
Now suppose a third type of agent. This agent has savings $b_3 = \left[4.1, 5.1, 7.1, 4.5, 0.9\right]$.  Use `numpy.vstack` to stack `b` from {numref}`ExerNumpy-array` on top of `b_3` to create a new $3\times 5$ matrix `b_new`.
```{exercise-end}
```

(SecNumPyFootnotes)=
## Footnotes

The footnotes from this chapter.

[^NumPy]: The website for NumPy is https://numpy.org.

[^Oliphant]: Travis Oliphant has a [Wikipedia page](https://en.wikipedia.org/wiki/Travis_Oliphant) {cite}`OliphantWiki`. We highly recommend [Oliphant's interview](https://youtu.be/gFEE3w7F0ww?si=XKcRlcw7FXkA9oxB) on the Lex Fridman Podcast from September 22, 2021 {cite}`Fridman:2021`.
