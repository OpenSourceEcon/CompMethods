
(Chap_StructEstIntro)=
# Introduction to Structural Estimation

> ``You keep using that word. I do not think it means what you think it means." Inigo Montoya, *The Princess Bride*

The term ``structural estimation" has been the source of debate in the economics profession. [TODO: Insert some of the debate here.]

The material for the chapters in this Structural Estimation section was initially developed in the Structural Estimation course I taught in the Masters in Computational Social Science program at the University of Chicago from 2017 to 2020.[^MACSScourses]

A good place to start in describing structural estimation is the definition of a {term}`model`.

```{prf:definition} Model
:label: DefStructEst_Model

A **model** is a set of cause and effect mathematical relationships, often specified with parameters $\theta$, among data $x$ or $(x,y)$ used to understand, explain, and predict phenomena. A model might be specified as,
\begin{equation*}
  g(x,\theta) = 0 \quad\text{or}\quad y = g(x,\theta)
\end{equation*}
where $g$ is a function or vector of functions that represents the mathematical relationships between variables and parameters.
```

```{prf:definition} Exogenous variables
:label: DefStructEst_ExogVar

**Exogenous variables** are inputs to the model, taken as given, or from outside the model. These can include both data $x$ and parameters $\theta$.
```

```{prf:definition} Endogenous variables
:label: DefStructEst_EndogVar

* **Endogenous variables** are outputs of the model or dependent on exogenous variables. These can include portions of the data $x$, sometimes designated as $y$ as in $y = g(x,\theta)$.
```

```{prf:definition} Data generating process (DGP)
:label: DefStructEst_DGP

The broadest definition of a **data generating process** (DGP) is a complete description of the mechanism that causes some observed phenomenon with all its dependencies. Unfortunately, in most realistic systems, this definition is too complex. A more practical definition of a **data generating process** is a simplified version of the process that causes some observed phenomenon with its key dependencies. The concept of a DGP is very similar to the concept of a {term}`model` from {prf:ref}`DefStructEst_Model`. A key characteristic of a DGP is that it must be specified in such as way that it could be used to simulate data.
```

```{prf:definition} Structural model
:label: DefStructEst_StructMod

A **structural model** in economics is a model in which the mathematical relationships among variables and parameters are derived from individuals', firms', or other organizations' optimization. These are often referred to as behavioral equations. **Structural models** can include linear models and linear approximations. But most often, **structural models** are nonlinear and dynamic.
```

```{prf:definition} Reduced form model
:label: DefStructEst_ReducedMod

A **reduced form model** in economics is a model in which the equations are either not derived from behavioral equations or are only implicitly a linear approximation of some more complicated model. However, because they are atheoretical and often nonparametric, machine learning models can be categorized as reduced form. **Reduced form models** are most often static, although time series econometric models are categorized as reduced form.
```

```{prf:definition} Structural estimation
:label: DefStructEst_StructEst

Put definition of **structural estimation** here.
```

```{prf:definition} Calibration
:label: DefStructEst_Calib

Put definition of **calibration** here.
```

```{prf:definition} Reduced form estimation
:label: DefStructEst_ReducedEst

Put definition of **reduced form estimation** here.
```

(SecStructEstIntroTypes)=
## Different types of models
A good introduction to structural estimation is to compare it to other types of research designs. {numref}`ExercStructEst_CompPaper` asks you to compare the structural approach to the reduced form approach. The following are some prominent research designs in economics, only some of which are structural estimations.

**Structural estimation papers**
* (Classic structural estimation) {cite}`Rust:1987`
* {cite}`BarskySims:2012`

**Reduced form estimation papers**
* {cite}`BaileyEtAl:2019`
* (Theory and reduced form and randomized controlled trial (RCT)) {cite}`AttanasioEtAl:2020`

**Theory**
* {cite}`StraubWerning:2020`


(SecStructEstIntroExerc)=
## Exercises

```{exercise} Persuasive short paper on structural estimation
:label: ExercStructEst_CompPaper
:class: green

**Persuasive short paper supporting either structural estimation or reduced form estimation or both.**
* Read {cite}`Keane:2010` and {cite}`Rust:2010`.
* Write a short persuasive paper of about one page (maximum of 1.5 pages) in which you make your case for either structural estimation or reduced form estimation or both. Note that both Keane and Rust are biased toward structural estimation.
* Make sure that you cite arguments that they use as evidence for or against your thesis.
* Refute (or temper) at least one of their arguments.
```


(SecStructEstIntroFootnotes)=
## Footnotes

The footnotes from this chapter.

[^MACSScourses]: I taught a course, entitled Structural Estimation, to graduate students, with a few advanced undergradutates, in the Masters in Computational Social Science program at the University of Chicago four times from 2017 to 2020. The content of each course is in the following GitHub repositories, with syllabi, lecture slides, Jupyter notebooks, tests, and problem sets: [Winter 2017](https://github.com/rickecon/StructEst_W17), [Winter 2018](https://github.com/rickecon/StructEst_W18), [Winter 2019](https://github.com/rickecon/StructEst_W19), and [Winter 2020](https://github.com/rickecon/StructEst_W20).
