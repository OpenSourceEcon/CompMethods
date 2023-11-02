---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(Chap_BasicEmpirMethods)=
# Basic Empirical Methods

The focus of this chapter is to give the reader a basic introduction to the standard empirical methods in data science, policy analysis, and economics. I want each reader to come away from this chapter with the following basic skills:

* Difference between **correlation** and **causation**
* Standard **data description**
* Basic understanding of **linear regression**
    * What do regression **coefficients** mean?
    * What do **standard errors** mean?
    * How can I estimate my own linear regression with standard errors?
    * Basic extensions: cross terms, quadratic terms, difference-in-difference
* Ideas behind bigger extensions of linear regression
    * Instrumental variables (omitted variable bias)
    * Logistic regression
    * Multiple equation models
    * Panel data
    * Time series data
    * Vector autoregression


In the next chapter {ref}`Chap_BasicMLintro`, I give a more detailed treatment of logistic regression as a bridge to learning the basics of machine learning.

Some other good resources on the topic of learning the basics of linear regression in Python include the [QuantEcon.org](https://quantecon.org/) lectures "[Simple Linear Regression Model](https://intro.quantecon.org/simple_linear_regression.html)" {cite}`SargentStachurski:2023a`, and "[Linear Regression in Python](https://python.quantecon.org/ols.html)" {cite}`SargentStachurski:2023b`.


(SecBasicEmpLit)=
## Basic Empirical Methods in the Literature

What are the standard empirical methods in the current version of the *American Economic Review* ([Vol. 113, No. 10, October 2023](https://www.aeaweb.org/issues/736))?

Allen, Bertazzini, and Heldring, "The Economic Origins of Government" {cite}`AllenEtAl:2023`
* Table 1, descriptive/summary statistics of the data
* Eq. 1: Difference-in-difference
\begin{equation*}
  Y_{c,t} = \sum_{k=0}^{-4}\left(\beta_k^{trmt}\times\mathbf{1}_k\times treated_c\right) + \rho_c + \gamma_t + \nu_{c,t} + \varepsilon_{c,t}
\end{equation*}
* Table 2, estimated coefficients, cross terms, standard errors

The iframe below contains a PDF of {cite}`AllenEtAl:2023` "The Economic Origins of Government".

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1ZR8a6DmbMbrW3K4Hdkgg5x4X1oFy_yNn/preview?usp=sharing">
  </iframe>
</div>


(SecBasicEmpCorrCaus)=
## Correlation versus Causation

What is the difference between correlation and causation?
* What are some examples of things that are correlated but do not "cause" each other?

What are some principles that cause correlation to not be causation?
* Third variable problem/omitted variable/spurious correlation
* Directionality/endogeneity

How do we determine causation?
* Randomized controlled trials (RCT)
* Laboratory experiments
* Natural experiments
* Quasi natural experiments


(SecBasicEmpDescr)=
## Data Description

Any paper that uses data needs to spend some ink summarizing and describing the data. This is usually done in tables. But it can also be done in cross tabulation, which is descriptive statistics by category. The most common types of descriptive statistics are the following:

* mean
* median
* variance
* count
* max
* min

Let's download some data, and read it in using the Pandas library for Python.[^PandasRef] The following example is adapted from QuantEcon's "[Linear Regression in Python](https://python.quantecon.org/ols.html)" lecture {cite}`SargentStachurski:2023b`.

The research question of the paper "The Colonial Origins of Comparative Development: An Empirical Investigation" {cite}`AcemogluEtAl:2001` is to determine whether or not differences in institutions can help to explain observed economic outcomes. How do we measure institutional differences and economic outcomes? In this paper:
* economic outcomes are proxied by log GDP per capita in 1995, adjusted for exchange rates,
* institutional differences are proxied by an index of protection against expropriation on average over 1985-95, constructed by the [Political Risk Serivces Group](https://www.prsgroup.com/).

These variables and other data used in the paper are available for download on [Daron Acemoglu’s webpage](https://economics.mit.edu/faculty/acemoglu/data/ajr2001).


(SecBasicEmpDescrBasic)=
### Basic Data Description

The following cells downloads the data from {cite}`AcemogluEtAl:2001` from the file `maketable1.dta` and displays the first five observations from the data.

```{code-cell} ipython3
:tags: []

import pandas as pd

df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/' +
                    'raw/master/ols/maketable1.dta')
```

The [`pandas.DataFrame.head`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) method returns the first $n$ forws of a DataFrame with column headings and index numbers. The default is `n=5`.

```{code-cell} ipython3
:tags: []

df1.head()
```

How many observations are in this dataset? What are the different countries in this dataset?

```{code-cell} ipython3
:tags: []

print("The number of observations (rows) in the dataset is:", df1.size)
print("")
print("A list of all the", len(df1["shortnam"].unique()),
      'unique countries in the "shortnam" variable is:')
print(df1["shortnam"].unique())
```

Pandas DataFrames have a built-in method `.describe()` that will give the basic descriptive statistics for the numerical variables of a dataset.

```{code-cell} ipython3
:tags: []

df1.describe()
```


<!-- {numref}`ExerBasicEmpir_MultLinRegress` -->


(SecBasicEmpirExercises)=
## Exercises

```{exercise-start} Multiple linear regression
:label: ExerBasicEmpir_MultLinRegress
:class: green
```
For this problem, you will use the 397 observations from the [`Auto.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/basic_empirics/Auto.csv) dataset in the [`/data/basic_empirics/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/basic_empirics) folder of the repository for this book.[^Auto] This dataset includes 397 observations on miles per gallon (`mpg`), number of cylinders (`cylinders`), engine displacement (`displacement`), horsepower (`horsepower`), vehicle weight (`weight`), acceleration (`acceleration`), vehicle year (`year`), vehicle origin (`origin`), and vehicle name (`name`).
1. Import the data using the [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) function. Look for characters that seem out of place that might indicate missing values. Replace them with missing values using the `na_values=...` option.
2. Produce a scatterplot matrix which includes all of the quantitative variables `mpg`, `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `year`, `origin`. Call your DataFrame of quantitative variables `df_quant`. [Use the pandas scatterplot function in the code block below.]
```python
from pandas.plotting import scatter_matrix

scatter_matrix(df_quant, alpha=0.3, figsize=(6, 6), diagonal='kde')
```
3. Compute the correlation matrix for the quantitative variables ($8\times 8$) using the [`pandas.DataFrame.corr()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) method.
4. Estimate the following multiple linear regression model of $mpg_i$ on all other quantitative variables, where $u_i$ is an error term for each observation, using Python's `statsmodels.api.OLS()` function.
    \begin{equation*}
      \begin{split}
        mpg_i &= \beta_0 + \beta_1 cylinders_i + \beta_2 displacement_i + \beta_3 horsepower_i + ... \\
        &\qquad \beta_4 weight_i + \beta_5 acceleration_i + \beta_6 year_i + \beta_7 origin_i + u_i
      \end{split}
    \end{equation*}
    * Which of the coefficients is statistically significant at the 1\% level?
    * Which of the coefficients is NOT statistically significant at the 10\% level?
    * Give an interpretation in words of the estimated coefficient $\hat{\beta}_6$ on $year_i$ using the estimated value of $\hat{\beta}_6$.
5. Looking at your scatterplot matrix from part (2), what are the three variables that look most likely to have a nonlinear relationship with $mpg_i$?
    * Estimate a new multiple regression model by OLS in which you include squared terms on the three variables you identified as having a nonlinear relationship to $mpg_i$ as well as a squared term on $acceleration_i$.
    * Report your adjusted R-squared statistic. Is it better or worse than the adjusted R-squared from part (4)?
    * What happened to the statistical significance of the $displacement_i$ variable coefficient and the coefficient on its squared term?
    * What happened to the statistical significance of the cylinders variable?
6. Using the regression model from part (5) and the `.predict()` function, what would be the predicted miles per gallon $mpg$ of a car with 6 cylinders, displacement of 200, horsepower of 100, a weight of 3,100, acceleration of 15.1, model year of 1999, and origin of 1?
```{exercise-end}
```


(SecBasicEmpirFootnotes)=
## Footnotes

The footnotes from this chapter.

[^PandasRef]: For a tutorial on using Python's Pandas package, see the {ref}`Chap_Pandas` chapter of this online book.

[^Auto]: The [`Auto.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/basic_empirics/Auto.csv) dataset comes from {cite}`JamesEtAl:2017` (ch. 3) and is also available at http://www-bcf.usc.edu/~gareth/ISL/data.html.
