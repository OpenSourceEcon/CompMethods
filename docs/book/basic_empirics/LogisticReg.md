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

(Chap_LogIntro)=
# Logistic Regression Model

This chapter has an executable [Google Colab notebook](https://colab.research.google.com/drive/1kNMOMvoKzuzNq_rw1yz86B3N98hgTaZ8?usp=sharing) with all the same code, data references, and images. The Google Colab notebook allows you to execute the code in this chapter in the cloud so you don't have to download Python, any of its packages, or any data to your local computer. You could manipulate and execute this notebook on any device with a browser, whether than be your computer, phone, or tablet.

The focus of this chapter is to give the reader a basic introduction to the logistic regression model, where it comes from, and how it can be interpreted.


(Sec_LogQuantQual)=
## Quantitative versus Qualitative Data
The linear regression models of chapter {ref}`Chap_BasicEmpirMethods` have continuous quantitative variables as dependent variables. That is, the $y_i$ variable takes on a continuum of values. We use a different class of models to estimate the relationship of exogenous variables to *qualitative* or *categorical*  or *discrete* endogenous or dependent variables.

Examples of qualitative or categorical variables include:

* Binary variables take on two values ($J=2$), most often 0 or 1. Examples: Male or female, dead or alive, accept or reject.
* General categorical variables can take on more than two values ($J\geq 2$). Examples: red, blue, or green; teenager, young adult, middle aged, senior.

Note with general categorical variables that order and numerical distance do not matter. As an example let $FlowerColor_i=\{red=1, blue=2,green=3\}$ be a function of $neighborhood_i$, $season_i$, and $income_i$.

$$ FlowerColor_i = \beta_0 + \beta_1 neighborhood_i + \beta_2 season_i + \beta_3 income_i + u_i $$

We could mathematically estimate this regression model, but would that make sense? What would be wrong with a regression model?


(Sec_LogQuantQualClassSet)=
### The classification setting
Let $y_i$ be a qualitative dependent variable on $N$ observations with $i$ being the index of the observation. Each observation $y_i$ can take on one of $J$ discrete values $j\in\{1,2,...J\}$. Let $x_{p,i}$ be the $i$th observation of the $p$th explanatory variable (independent variable) such that $X_i=\{x_{1,i}, x_{2,i}, ... x_{P,i}\}$. Then the general formulation of a classifier comes in the following two forms,

```{math}
    :label: EqLog_GenClassModel
    Pr(y_i=j|X_i,\theta) = f(X_i|\theta) \quad\forall i, j \quad\text{or}\quad \sum_{j=1}^J I_j(y_i=j) = f(X_i|\theta) \quad\forall i, j
```

where $I_j$ in the second formulation is an indicator function that equals 1 when $y_i=j$ and equals 0 otherwise.


(Sec_LogRegClass)=
## Logistic Regression Classifier
In this section, we will look at two models for binary (0 or 1) categorical dependent variables. We describe the first model--the linear probability (LP) model--for purely illustrative purposes. This is because the LP model has some serious shortcomings that make it almost strictly dominated by our second model in this section.

The second model--the logistic regression (logit, binary classifier) model--will be the focus of this section. There is another variant of this model, the probit model. But the logistic model is the more flexible, more easily interpretable, and more commonly used of the two.


(Sec_LogLPM)=
### The linear probability (LP) model

One option in which a regression is barely acceptable for modeling a binary (categorical) dependent variable is the linear probability (LP) model. When the dependent variable has only two categories, it can be modeled as $y_i\in\{0,1\}$ without loss of generality. Let the variable $z_i$ be interpreted as the probability that $y_i=1$ given the data $X_i$ and parameter values $\theta=\{\beta_0,\beta_1,...\beta_P\}$.

```{math}
    :label: EqLog_LPM
    z_i = Pr(y_i=1|X_i,\theta) = \beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i} + ... \beta_P x_{P,i} + u_i
```

The LP model can be a nice, easy, computationally convenient way to estimate the probability of outcome $y_i=1$. This could also be reinterpreted, without loss of generality, as the probability that $y_i=0$. This is equivalent to a redefinition of which outcome is defined as $y_i=1$.

The main drawback of the LP model is that the predicted values of the probability that $y_i=1$ or $Pr(y_i=1|X_i,\theta)$ can be greater than 1 and can be less than 0. It is for this reason that it is very difficult to publish any research based on an LP model.


(Sec_LogLogit)=
### The logistic (logit) regression classifier

In contrast to the linear probability model, a good classifier tranforms numerical values from explanatory variables or feature variables into a probability that is strictly between 0 and 1. More specifically this function must take any numer on the real line between $-\infty$ and $\infty$ and map it to the $[0,1]$ interval. In addition, we want a monotonically increasing relationship between $x$ and the function $f(x)$. What are some functions with this property? Candidates include the following functions.

* $f(x)=\text{max}\Bigl(0, \,\text{min}\bigl(1, x\bigr)\Bigr)$
* $f(x)=\frac{e^x}{1 + e^x}$
* $f(x) = \arctan(x)$
* $f(x) = \text{cdf}(x)$

Why don't functions like $\sin(x)$, $\cos(x)$, and $\frac{|x|}{1+|x|}$ fit these criteria?

The second function in the bulletted list above is the logistic function. The logistic regression model is a binary dependent variable classifier that constrains its predicted values to be stricly between 0 and 1. The logistic function is the following,

```{math}
    :label: EqLog_Logistic
    f(x) = \frac{e^x}{1 + e^x} \quad\forall x
```

and has the following general shape.

```{code-cell} ipython3
:tags: ["hide-input", "remove_output"]

import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(-6, 6, 500)
y_vals = np.exp(x_vals) / (1 + np.exp(x_vals))
plt.plot(x_vals, y_vals, color="blue")
plt.scatter(0, 0.5, color="black", s=15)
plt.title(r"Logistic function for $x\in[-6,6]$")
plt.xlabel(r'$x$ values')
plt.ylabel(r'$f(x)$ values')
plt.grid(color='gray', linestyle=':', linewidth=1, alpha=0.5)
plt.show()
```

```{figure} ../../../images/basic_empirics/logit/logit_gen.png
:height: 500px
:name: FigLogit_logit_gen

Logistic function for $x\in[-6,6]$
```

The logistic regression function is the specific case of the logistic function where the value of $x$ in the general logistic function {eq}`EqLog_Logistic` is replaced by a linear combination of variables $\beta_0 + \beta_1 x_{1,i} + ...\beta_P x_{P,i}$ similar to a linear regression model.

```{math}
    :label: EqLog_Logit_std
    Pr(y_i=1|X_i,\theta) = \frac{e^{X_i\beta}}{1 + e^{X_i\beta}} = \frac{e^{\beta_0 + \beta_1 x_{1,i} + ...\beta_P x_{P,i}}}{1 + e^{\beta_0 + \beta_1 x_{1,i} + ...\beta_P x_{P,i}}}
```

or equivalently

```{math}
    :label: EqLog_Logit_neg
    Pr(y_i=1|X_i,\theta) = \frac{1}{1 + e^{-X_i\beta}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{1,i} + ...\beta_P x_{P,i})}}
```

We could estimate the paramters $\theta=\{\beta_0,\beta_1,...\beta_P\}$ by generalized method of moments (GMM) using nonlinear least squares or a more general set of moments to match.[^GMM] But maximum likelihood estimation is the most common method for estimating the parameters $\theta$ because of its more robust statistical properties.[^MaxLikeli] Also, the distributional assumptions are built into the model, so they are not overly strong.


(Sec_LogLogitNLLS)=
#### Nonlinear least squares estimation
If we define $z_i = Pr(y_i=1|X_i,\theta)$, then the error in the logistic regression is the following.

```{math}
    :label: EqLog_LogitNLLS_err
    \varepsilon_i = y_i - z_i
```

The GMM specification of the nonlinear least squares method of estimating the parameter vector $\theta$ would then be the following.[^GMM]

```{math}
    :label: EqLog_LogitNLLS_gmm
    \begin{split}
      \hat{\theta}_{nlls} = \theta:\quad &\min_{\theta} \sum_{i=1}^N\varepsilon_i^2 \quad = \quad \min_{\theta}\sum_{i=1}^N\bigl(y_i - z_i \bigr)^2 \quad \\
      &= \quad \min_{\theta} \sum_{i=1}^N\Bigl[y_i - Pr(y_i=1|X_i,\theta)\Bigr]^2
    \end{split}
```


(Sec_LogLogitMLE)=
#### Maximum likelihood estimation
We characterized the general likelihood function for a sample of data as the probability that the given sample $(y_i,X_i)$ came from the assumed distribution given parameter values $Pr(y_i=1|X_i,\theta)$.

```{math}
    :label: EqLog_LogitMLE_like
    \mathcal{L}(y_i,X_i|\theta) = \prod_{i=1}^N Pr(y_i=1|X_i,\theta)^{y_i}\bigl[1 - Pr(y_i=1|X_i,\theta)\bigr]^{1 - y_i}
```

The intuition of this likelihood function is that you want the probability of the observations for which $y_i=1$ to be close to one $Pr(X)$, and you want the probability of the observations for which $y_i=0$ to also be close to one $1 - Pr(X)$.

The log-likelihood function, which the MLE problem maximizes is the following.

```{math}
    :label: EqLog_LogitMLE_loglike
    \ln\bigl[\mathcal{L}(y_i,X_i|\theta)\bigr] = \sum_{i=1}^N\Bigl(y_i\ln\bigl[Pr(y_i=1|X_i,\theta)\bigr] + (1 - y_i)\ln\bigl[1 - Pr(y_i=1|X_i,\theta)\bigr]\Bigr)
```

The MLE problem for estimating $\theta$ of the logistic regression model is, therefore, the following.[^MaxLikeli]

```{math}
    :label: EqLog_LogitMLE_maxprob
    \hat{\theta}_{mle} = \theta:\quad \max_{\theta} \ln\bigl[\mathcal{L}(y_i,X_i|\theta)\bigr]
```

(Sec_LogLogitTitanic)=
#### Titanic example
A good example of logistic regression comes from a number of sources. But I am adapting some code and commentary from [http://www.data-mania.com/blog/logistic-regression-example-in-python/](http://www.data-mania.com/blog/logistic-regression-example-in-python/). The research question is to use a famous Titanic passenger dataset to try to identify the characteristics that most predict whether you survived $y_i=1$ or died $y_i=0$.

```{code-cell} ipython3
:tags: []

import pandas as pd

url = ('https://raw.githubusercontent.com/OpenSourceEcon/CompMethods/' +
      'main/data/basic_empirics/logit/titanic-train.csv')
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                   'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
titanic.describe()
```

The variable descriptions are the following:
* `Survived`: Survival (0 = No; 1 = Yes)
* `Pclass`: Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
* `Name`: Name
* `Sex`: Gender
* `Age`: Age
* `SibSp`: Number of siblings/spouses aboard
* `Parch`: Number of parents/children aboard
* `Ticket`: Ticket number
* `Fare`: Passenger fare (British pound)
* `Cabin`: Cabin
* `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Let's first check that our target variable, `Survived`, is binary. Since we are building a model to predict survival of passangers from the Titanic, our target is going to be the `Survived` variable from the titanic dataframe. To make sure that it is a binary variable, let's use Seaborn's `countplot()` function.

```{code-cell} ipython3
:tags: []

titanic['Survived'].value_counts()
```


(SecLogFootnotes)=
## Footnotes

The footnotes from this chapter.

[^GMM]: See the {ref}`Chap_GMM` chapter of this book.

[^MaxLikeli]: See the {ref}`Chap_MaxLikeli` chapter of this book.
