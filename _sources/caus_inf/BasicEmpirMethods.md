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

Put basic empirical methods here. {numref}`ExerBasicEmpir_MultLinRegress`


(SecBasicEmpirExercises)=
## Exercises

```{exercise-start} Multiple linear regression
:label: ExerBasicEmpir_MultLinRegress
:class: green
```
For this problem, you will use the 397 observations from the [`Auto.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/BasicEmpirMethods/Auto.csv) dataset in the [`/data/BasicEmpirMethods/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/BasicEmpirMethods) folder of the repository for this book.[^Auto] This dataset includes 397 observations on miles per gallon (`mpg`), number of cylinders (`cylinders`), engine displacement (`displacement`), horsepower (`horsepower`), vehicle weight (`weight`), acceleration (`acceleration`), vehicle year (`year`), vehicle origin (`origin`), and vehicle name (`name`).
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

[^Auto]: The [`Auto.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/BasicEmpirMethods/Auto.csv) dataset comes from {cite}`JamesEtAl:2017` (ch. 3) and is also available at http://www-bcf.usc.edu/~gareth/ISL/data.html.
