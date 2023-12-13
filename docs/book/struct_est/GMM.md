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

(Chap_GMM)=
# Generalized Method of Moments Estimation

This chapter describes the generalized method of moments (GMM) estimation method. All data and images from this chapter can be found in the data directory ([./data/gmm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/gmm/)) and images directory ([./images/gmm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/images/gmm/)) for the GitHub repository for this online book.


(SecGMM_GMMvMLE)=
## GMM vs. MLE: Strengths and weaknesses

A paper by {cite}`FuhrerEtAl:1995` studies the accuracy and efficiency of the maximum likelihood (ML) estimator versus the generalized method of moments (GMM) estimator in the context of a simple linear-quadratic inventory model. They find that ML has some very nice properties over GMM in small samples when the model is simple. In the spirit of the {cite}`FuhrerEtAl:1995` paper, we list the strengths and weaknesses of MLE vs. GMM more generally. I recommend you read the introduction to {cite}`FuhrerEtAl:1995`. This paper provides big support for maximum likelihood estimation over generalized method of moments. However, GMM estimation allows for less strong assumptions.
* GMM almost always rejects the model (Hansen J-test)
* MLE supports the model, kind of by assumption
* "Monte Carlo experiments reveal that the GMM estimates are often biased (apparently due to poor instruments), statistically insignificant, economically implausible, and dynamically unstable."
* "The ML estimates are generally unbiased (even in misspecifipd models), statistically significant, economically plausible, and dynamically stable."
* "Asymptotic standard errors for ML are 3 to 15 times smaller than for GMM."


(SecGMM_MLEstr)=
### MLE strengths

* More statistical significance. In general, MLE provides more statistical significance for parameter estimates than does GMM. This comes from the strong distributional assumptions that are necessary for the ML estimates.
* ML estimates are less sensitive to parameter or model normalizations than are GMM estimates.
* ML estimates have nice small sample properties. ML estimates have less bias and more efficiency with small data samples than GMM estimates in many cases.


(SecGMM_MLEwk)=
### MLE weaknesses

* MLE requires strong distributional assumptions. For MLE, the data generating process (DGP) must be completely specified. This assumes a lot of knowledge about the DGP. This assumption is likely almost always wrong.
* MLE is very difficult in rational expectations models. This is because the consistency of beliefs induces a nonlinearity in the likelihood function that makes it difficult to find the global optimum.
* MLE is very difficult in nonlinear models. The likelihood function can become highly nonlinear in MLE even if the model is linear when the data are irregular. This difficulty is multiplied when the model itself is more complicated and nonlinear.


(SecGMM_GMMstr)=
### GMM strengths

* GMM allows for most flexible identification. GMM estimates can be identified by any set of moments from the data as long as you have at least as many moments as you have parameters to estimate and that those moments are independent enough to identify the parameters. (And the parameters are independent enough of each other to be separately identified.)
* Good large sample properties. The GMM estimator is strongly consistent and asymptotically normal. GMM will likely be the best estimator if you have a lot of data.
* GMM requires minimal assumptions about the DGP. In GMM, you need not specify the distributions of the error terms in your model of the DGP. This is often a strength, given that most error are not observed and most models are gross approximations of the true DGP.


(SecGMM_GMMwk)=
### GMM weaknesses

* GMM estimates are usually less statistically significant than ML estimates. This comes from the minimal distributional assumptions. GMM parameter estimates usually are measured with more error.
* GMM estimates can be sensitive to normalizations of the model or parameters.
* GMM estimates have bad small sample properties. GMM estimates can have large bias and inefficiency in small samples.


(SecGMM_keyqst)=
### Key questions when deciding between MLE and GMM

* How much data is available for the estimation? Large data samples will make GMM relatively more attractive than MLE because of the nice large sample properties of GMM and fewer required assumptions on the model.
* How complex is the model? Linear models or quadratic models are much easier to do using MLE than are more highly nonlinear models. Rational expectations models (macroeconomics) create an even more difficult level of nonlinearity that pushes you toward GMM estimation.
* How comfortable are you making strong distributional assumptions? MLE requires a complete specification of all distributional assumptions of the model DGP. If you think these assumptions are too strong, you should use GMM.


(SecGMM_GMMest)=
## The GMM estimator

GMM was first formalized by {cite}`Hansen:1982`. A strength of GMM estimation is that the econometrician can remain completely agnostic as to the distribution of the random variables in the DGP. For identification, the econometrician simply needs at least as many moment conditions from the data as he has parameters to estimate.

A *moment* of the data is broadly defined as any statistic that summarizes the data to some degree. A data moment could be as narrow as an individual observation from the data or as broad as the sample average. GMM estimates the parameters of a model or data generating process to make the model moments as close as possible to the corresponding data moments. See {cite}`DavidsonMacKinnon:2004`, chapter 9 for a more detailed treatment of GMM. The estimation methods of linear least squares, nonlinear least squares, generalized least squares, and instrumental variables estimation are all specific cases of the more general GMM estimation method.

Let $m(x)$ be an $R\times 1$ vector of moments from the real world data $x$, where $m_r(x)$ is the $r$th data moment. And let $x$ be an $N\times K$ matrix of data with $K$ columns representing $K$ variables and $N$ observations.

```{math}
    :label: EqGMM_GMMest_datamomvec
    m(x) \equiv \left[m_1(x), m_2(x), ...m_R(x)\right]^T
```

Let the model DGP be characterized as $F(x,\theta)=0$, where $F$ is a vector of equations, each of which is a function of the data $x$ and the $K\times 1$ parameter vector $\theta$. Then define $m(x|\theta)$ as a vector of $R$ moments from the model that correspond to the real-world moment vector $m(x)$, where $m_r(x|\theta)$ is the $r$th model moment.

```{math}
    :label: EqGMM_GMMest_modmomvec
    m(x|\theta) \equiv \left[m_1(x|\theta), m_2(x|\theta), ...m_R(x|\theta)\right]^T
```

Note that GMM requires both real world data $x$ and moments that can be calculated from both the data $m(x)$ and from the model $m(x|\theta)$ in order to estimate the parameter vector $\hat{\theta}_{GMM}$. There is also a stochastic way to generate moments from the model, which we discuss later in our section on Simulated Method of Moments (SMM).

The GMM approach of estimating the parameter vector $\hat{\theta}_{GMM}$ is to choose $\theta$ to minimize some distance measure of the model moments $m(x|\theta)$ from the data moments $m(x)$.

```{math}
    :label: EqGMM_GMMest_genprob
    \hat{\theta}_{GMM}=\theta:\quad \min_{\theta}\: ||m(x|\theta) - m(x)||
```

The distance measure $||m(x|\theta) - m(x)||$ can be any kind of norm. But it is important to recognize that your estimates $\hat{\theta}_{GMM}$ will be dependent on what distance measure (norm) you choose. The most widely studied and used distance metric in GMM estimation is the $L^2$ norm or the sum of squared errors in moments. Define the moment error function $e(x|\theta)$ as the $R \times 1$ vector of either the percent difference in the vector of model moments from the data moments or the simple difference.

```{math}
    :label: EqGMM_GMMest_momerr
    e(x|\theta) \equiv \frac{m(x|\theta) - m(x)}{m(x)} \quad\text{or}\quad e(x|\theta) \equiv m(x|\theta) - m(x)
```

It is important when possible that the error function $e(x|\theta)$ be a percent deviation of the moments (given that none of the data moments are 0). This puts all the moments in the same units, which helps make sure that no moments receive unintended weighting simply due to their units. This ensures that the problem is scaled properly and does not suffer from ill conditioning. However, percent deviations become computationally problematic when the data moments are zero or close to zero. In that case, you would use a simple difference.

The GMM estimator is the following,

```{math}
    :label: EqGMM_GMMest_qdrprob
    \hat{\theta}_{GMM}=\theta:\quad \min_{\theta}\:e(x|\theta)^T \, W \, e(x|\theta)
```

where $W$ is an $R\times R$ weighting matrix in the criterion function. For now, think of this weighting matrix as the identity matrix. But we will show in Section {ref}`SecGMM_Wgt` a more optimal weighting matrix. We call the quadratic form expression $e(x|\theta)^T \, W \, e(x|\theta)$ the *criterion function* because it is a strictly positive scalar that is the object of the minimization in the GMM problem in the general statement of the problem {eq}`EqGMM_GMMest_genprob` and in the sum of squared errors version of the problem {eq}`EqGMM_GMMest_qdrprob`. The $R\times R$ weighting matrix $W$ in the criterion function allows the econometrician to control how each moment is weighted in the minimization problem. For example, an $R\times R$ identity matrix for $W$ would give each moment equal weighting of 1, and the criterion function would be a simply sum of squared percent deviations (errors). Other weighting strategies can be dictated by the nature of the problem or model.


(SecGMM_Wgt)=
## The weighting matrix (W)

In the GMM criterion function in the problem statement {eq}`EqGMM_GMMest_qdrprob`, some moment weighting matrices $W$ produce precise estimates while others produce poor estimates with large variances. We want to choose the optimal weighting matrix $W$ with the smallest possible asymptotic variance. This is an efficient optimal GMM estimator. The optimal weighting matrix is the inverse variance covariance matrix of the moments at the optimal parameter values,

```{math}
    :label: EqGMM_Wgt_gen
    W^{opt} \equiv \Omega^{-1}(x|\hat{\theta}_{GMM})
```

where $\Omega(x|\theta)$ is the variance covariance matrix of the moment condition errors $E(x|\theta)$ from each observation in the data (to be defined below). The intuition for using the inverse variance covariance matrix $\Omega^{-1}$ as the optimal weighting matrix is the following. You want to downweight moments that have a high variance, and you want to weight more heavily the moments that are generated more precisely.

Notice that this definition of the optimal weighting matrix is circular. $W^{opt}$ is a function of the GMM estimates $\hat{\theta}_{GMM}$, but the optimal weighting matrix is used in the estimation of $\hat{\theta}_{GMM}$. This means that one has to use some kind of iterative fixed point method to find the true optimal weighting matrix $W^{opt}$. Below are some examples of weighting matrices to use.


(SecGMM_Wgt_I)=
### The identity matrix (W=I)

Many times, you can get away with just using the identity matrix as your weighting matrix $W = I$. This changes the criterion function to a simple sum of squared error functions such that each moment has the same weight.

```{math}
    :label: EqGMM_GMMest_WI
    \hat{\theta}_{GMM}=\theta:\quad \min_{\theta}\:e(x|\theta)^T \, e(x|\theta)
```

If the problem is well conditioned and well identified, then your GMM estimates $\hat{\theta}_{GMM}$ will not be greatly affected by this simplest of weighting matrices.


(SecGMM_Wgt_2step)=
### Two-step variance-covariance estimator of W

The most common method of estimating the optimal weighting matrix for GMM estimates is the two-step variance covariance estimator. The name "two-step" refers to the two steps used to get the weighting matrix.

The first step is to estimate the GMM parameter vector $\hat{\theta}_{1,GMM}$ using the simple identity matrix as the weighting matrix $W = I$ as in {eq}`EqGMM_GMMest_WI`.

```{math}
    :label: EqGMM_GMMest_2stp_1
    \hat{\theta}_{1, GMM}=\theta:\quad \min_{\theta}\:e(x|\theta)^T \, I \, e(x|\theta)
```

We use the $R\times 1$ moment error vector and the Step 1 GMM estimate $e(x|\hat{\theta}_{1,GMM})$ to get a new estimate of the variance-covariance matrix.

```{math}
    :label: EqGMM_GMMest_2stp_2VarCov
    \hat{\Omega}_2 = e(x|\hat{\theta}_{1,GMM})\,e(x|\hat{\theta}_{1,GMM})^T
```

This is simply saying that the $(r,s)$-element of the $R\times R$ estimator of the variance-covariance matrix of the moment vector is the following.

```{math}
    :label: EqGMM_2stepVarCov_rs
    \hat{\Omega}_{2,r,s} = \Bigl[m_r(x|\hat{\theta}_{1,GMM}) - m_{r}(x)\Bigr]\Bigl[m_s(x|\theta) - m_s(x)\Bigr]
```

The optimal weighting matrix is the inverse of the two-step variance covariance matrix.

```{math}
    :label: EqGMM_estW_2step
    \hat{W}^{two-step} \equiv \hat{\Omega}_2^{-1}
```

Lastly, re-estimate the GMM estimator using the optimal two-step weighting matrix $\hat{W}^{2step}$.

```{math}
    :label: EqGMM_theta_2step_2
    \hat{\theta}_{2,GMM}=\theta:\quad \min_{\theta}\:e(x|\theta)^T \, \hat{W}^{two-step} \, e(x|\theta)
```

$\hat{\theta}_{2,GMM}$ is called the two-step GMM estimator.


(SecGMM_W_iter)=
### Iterated variance-covariance estimator of W

The truly optimal weighting matrix $W^{opt}$ is the iterated variance-covariance estimator of $W$. This procedure is to just repeat the process described in the two-step GMM estimator until the estimated weighting matrix no longer significantly changes between iterations. Let $i$ index the $i$th iterated GMM estimator,

```{math}
    :label: EqGMM_theta_2step_i
    \hat{\theta}_{i, GMM}=\theta:\quad \min_{\theta}\:e(x|\theta)^T \, \hat{W}_{i} \, e(x|\theta)
```

and the $(i+1)$th estimate of the optimal weighting matrix is defined as the following.

```{math}
    :label: EqGMM_estW_istep
    \hat{W}_{i+1} \equiv \hat{\Omega}_{i+1}^{-1}\quad\text{where}\quad \hat{\Omega}_{i+1} = e(x|\hat{\theta}_{i,GMM})\,e(x|\hat{\theta}_{i,GMM})^T
```

The iterated GMM estimator $\hat{\theta}_{it,GMM}$ is the $\hat{\theta}_{i,GMM}$ such that $\hat{W}_{i+1}$ is very close to $\hat{W}_{i}$ for some distance metric (norm).

```{math}
    :label: EqGMM_theta_it
    \hat{\theta}_{it,GMM} = \hat{\theta}_{i,GMM}: \quad || \hat{W}_{i+1} - \hat{W}_{i} || < \varepsilon
```


(SecGMM_W_NW)=
### Newey-West consistent estimator of $\Omega$ and W

[TODO: Need to get this right for the GMM case.] The Newey-West estimator of the optimal weighting matrix and variance covariance matrix is consistent in the presence of heteroskedasticity and autocorrelation in the data (See {cite}`NeweyWest:1987`). {cite}`AddaCooper:2003` (p. 82) have a nice exposition of how to compute the Newey-West weighting matrix $\hat{W}_{nw}$. The asymptotic representation of the optimal weighting matrix $\hat{W}^{opt}$ is the following:

```{math}
    :label: EqGMM_estW_WhatOpt
    \hat{W}^{opt} = \lim_{N\rightarrow\infty}\frac{1}{N}\sum_{i=1}^N \sum_{l=-\infty}^\infty E(x_i|\theta)E(x_{i-l}|\theta)^T
```

The Newey-West consistent estimator of $\hat{W}^{opt}$ is:

```{math}
    :label: EqGMM_estW_NW
    \hat{W}_{nw} = \Gamma_{0,N} + \sum_{v=1}^q \left(1 - \left[\frac{v}{q+1}\right]\right)\left(\Gamma_{v,N} + \Gamma^T_{v,N}\right)
```

where

```{math}
    :label: EqGMM_estW_NWGamma
    \Gamma_{v,N} = \frac{1}{N}\sum_{i=v+1}^N E(x_i|\theta)E(x_{i-v}|\theta)^T
```

Of course, for autocorrelation, the subscript $i$ can be changed to $t$.


(SecGMM_VarCovTheta)=
## Variance-Covariance Estimator of $\hat{\theta}$

The estimated variance-covariance matrix $\hat{\Sigma}$ of the estimated parameter vector $\hat{\theta}_{GMM}$ is different from the variance-covariance matrix $\hat{\Omega}$ of the moment vector $e(x|\theta)$ from the previous section. $\hat{\Omega}$ from the previous section is the $R\times R$ variance-covariance matrix of the $R$ moment errors used to identify the $K$ parameters $\theta$ to be estimated. The estimated variance-covariance matrix of the estimated parameter vector $\hat{\Sigma}$ is a $K\times K$ matrix. We say the model is exactly identified if $K = R$. We say the model is overidentified if $K<R$.

Similar to the inverse Hessian estimator of the variance-covariance matrix of the maximum likelihood estimator from the {ref}`Chap_MLE`, the GMM variance-covariance matrix is related to the derivative of the criterion function with respect to each parameter. The intuition is that if the second derivative of the criterion function with respect to the parameters is large, there is a lot of curvature around the criterion minimizing estimate. In other words, the parameters of the model are precisely estimated. The inverse of the Hessian matrix will be small.

Define $R\times K$ matrix $d(x|\theta)$ as the Jacobian matrix of derivatives of the $R\times 1$ error vector $e(x|\theta)$.

```{math}
    :label: EqGMM_errvec_deriv
    \begin{equation}
    d(x|\theta) \equiv
        \begin{bmatrix}
        \frac{\partial e_1(x|\theta)}{\partial \theta_1} & \frac{\partial e_1(x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_1(x|\theta)}{\partial \theta_K} \\
        \frac{\partial e_2(x|\theta)}{\partial \theta_1} & \frac{\partial e_2(x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_2(x|\theta)}{\partial \theta_K} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial e_R(x|\theta)}{\partial \theta_1} & \frac{\partial e_R(x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_R(x|\theta)}{\partial \theta_K}
        \end{bmatrix}
    \end{equation}
```

The GMM estimates of the parameter vector $\hat{\theta}_{GMM}$ are assymptotically normal. If $\theta_0$ is the true value of the parameters, then the following holds,

```{math}
    :label: EqGMM_theta_plim
    \text{plim}_{N\rightarrow\infty}\sqrt{N}\left(\hat{\theta}_{GMM} - \theta_0\right) \sim \text{N}\left(0, \left[d(x|\theta)^T W d(x|\theta)\right]^{-1}\right)
```

where $W$ is the optimal weighting matrix from the GMM criterion function. The GMM estimator for the variance-covariance matrix $\hat{\Sigma}_{GMM}$ of the parameter vector $\hat{\theta}_{GMM}$ is the following.

```{math}
    :label: EqGMM_SigmaHat
    \hat{\Sigma}_{GMM} = \frac{1}{N}\left[d(x|\theta)^T W d(x|\theta)\right]^{-1}
```

In the examples below, we will use a finite difference method to compute numerical versions of the Jacobian matrix $d(\tilde{x},x|\theta)$. The following is a first-order forward finite difference numerical approximation of the first derivative of a function.

```{math}
    :label: EqGMM_finitediff_1
    f'(x_0) = \lim_{h\rightarrow 0} \frac{f(x_0 + h) - f(x_0)}{h}
```

The following is a centered second-order finite difference numerical approximation of the derivative of a function. (See [BYU ACME numerical differentiation lab](https://github.com/UC-MACSS/persp-model-econ_W19/blob/master/Notes/ACME_NumDiff.pdf) for more details.)

```{math}
    :label: EqGMM_finitediff_2
    f'(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h}
```


(SecGMM_Ex)=
## Examples

In this section, we will use GMM to estimate parameters of the models from the {ref}`Chap_MLE` chapter. We will also go through the standard moment conditions in most econometrics textbooks in which the conditional and unconditional expectations provide moments for estimation.


(SecGMM_Ex_Trunc)=
### Fitting a truncated normal to intermediate macroeconomics test scores

Let's revisit the problem from the {ref}`Chap_MLE` chapter of fitting a truncated normal distribution to intermediate macroeconomics test scores. The data are in the text file [`Econ381totpts.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/gmm/Econ381totpts.txt) in the GitHub repository [`../data/gmm/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/smm) folder for this executable book. Recall that these test scores are between 0 and 450. The figure below shows a histogram of the data, as well as three truncated normal PDF's. The black line is the ML estimate of $\mu$ and $\sigma$ of the truncated normal pdf. The red and the green lines are just the PDF's of two "arbitrarily" chosen combinations of the truncated normal parameters $\mu$ and $\sigma$.


(SecGMM_Ident)=
## Identification

An issue that we saw in the examples from Section {ref}`SecGMM_Ex` is that there is some science as well as some art in choosing moments to identify the parameters in a GMM estimation.

* The $\mu$ and $\sigma$ parameters were identified more precisely when using the two-step estimator of the optimal weighting matrix instead of the identity matrix.
* The overidentified four-moment model of total scores produced much smaller standard errors for both $\mu$ and $\sigma$ than did the two-moment model.

Suppose the parameter vector $\theta$ has $K$ elements, or rather, $K$ parameters to be estimated. In order to estimate $\theta$ by GMM, you must have at least as many moments as parameters to estimate $R\geq K$. If you have exactly as many moments as parameters to be estimated $R=K$, the model is said to be *exactly identified*. If you have more moments than parameters to be estimated $R>K$, the model is said to be *overidentified*. If you have fewer moments than parameters to be estimated $R<K$, the model is said to be *underidentified*. There are good reasons to overidentify $R>K$ the model in GMM estimation as we saw in the previous example. The main reason is that not all moments are orthogonal. That is, some moments convey roughly the same information about the data and, therefore, do not separately identify any extra parameters. So a good GMM model often is overidentified $R>K$.

One last point about GMM regards moment selection and verification of results. The real world has an infinite supply of potential moments that describe some part of the data. Choosing moments to estimate parameters by GMM requires understanding of the model, intuition about its connections to the real world, and artistry. A good GMM estimation will include moments that have some relation to or story about their connection to particular parameters of the model to be estimated. In addition, a good verification of a GMM estimation is to take some moment from the data that was not used in the estimation and see how well the corresponding moment from the estimated model matches that *outside moment*.


(SecGMM_LinReg)=
## Linear regression by GMM and relation to OLS


(SecGMM_LinReg_OLS)=
### Ordinary least squares: overidentification

The most common method of estimating the parameters of a linear regression is using the ordinary least squares (OLS) estimator. This estimator is just special type of generalized method of moments (GMM) estimator. A simple regression specification in which the dependent variable $y_i$ is a linear function of two independent variables $x_{1,i}$ and $x_{2,i}$ is the following:

```{math}
    :label: EqGMM_LinReg_LinReg
    y_i = \beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i} + \varepsilon_i \quad\text{where}\quad \varepsilon_i\sim N\left(0,\sigma^2\right)
```

Note that we can solve for the parameters $(\beta_0,\beta_1,\beta_2)$ in a number of ways. And we can do it without making any assumptions about the distribution of the error terms $\varepsilon_i$.

One way we might choose the parameters is to choose $(\beta_0,\beta_1,\beta_2)$ to minimize the distance between the $N$ observations of $y_i$ and the $N$ predicted values for $y_i$ given by $\beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i}$. You can think of the $N$ observations of $y_i$ as $N$ data moments. And you can think of the $N$ observations of $\beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i}$ as $N$ model moments. The least squares estimator minimizes the sum of squared errors, which is the sum of squared deviations between the $N$ values of $y_i$ and  $\beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i}$.

```{math}
    :label: EqGMM_LinReg_Errs
    \varepsilon_i = y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}
```

```{math}
    :label: EqGMM_LinReg_gmmprob
    \hat{\theta}_{OLS} = \theta:\quad \min_{\theta} \varepsilon^T\, I \, \varepsilon
```

The OLS GMM estimator of the linear regression model is an overidentified GMM estimator, in most cases, because the number of moments $R=N$ is greater than the number of parameters to be estimated $K$.

Let the $N\times 1$ vector of $y_i$'s be $Y$. Let the $N\times 3$ vector of data $(1, x_{1,i}, x_{2,i})$ be $X$. And let the vector of three parameters $(\beta_0, \beta_1, \beta_2)$ be $\beta$. It can be shown that the OLS estimator for the vector of parameters $\beta$ is the following.

```{math}
    :label: EqGMM_LinReg_xxxy
    \hat{\beta}_{OLS} = (X^T X)^{-1}(X^T Y)
```

But you could also just estimate the coefficients using the criterion function in the GMM statement of the problem above. This method is called nonlinear least squares or generalized least squares. Many applications of regression use a weighting matrix in the criterion function that adjusts for issues like heteroskedasticity and autocorrelation.

Lastly, many applications use a different distance metric than the weighted sum of squared errors for the difference in moments. Sum of squared errors puts a large penalty on big differences. Sometimes you might want to maximize the sum of absolute errors, which is sometimes called median regression. You could also minimize the maximum absolute difference in the errors, which is even more extreme than the sum of squared errors on penalizing large differences.


(SecGMM_LinReg_mom)=
### Linear regression by moment condition: exact identification

The exactly identified GMM approach to estimating the linear regression model comes from the underlying statistical assumptions of the model. We usually assume that the dependent variable $y_i$ and the independent variables $(x_{1,i}, x_{2,i})$ are not correlated with the error term $\varepsilon_i$. This implies the following three conditions.

```{math}
    :label: EqGMM_LinReg_momcond_y
    E\left[y^T \varepsilon\right] = 0
```

```{math}
    :label: EqGMM_LinReg_momcond_x1
    E\left[x_1^T \varepsilon\right] = 0
```

```{math}
    :label: EqGMM_LinReg_momcond_x2
    E\left[x_2^T \varepsilon\right] = 0
```

The data analogues for these moment conditions are the following.

```{math}
    :label: EqGMM_LinReg_datacond_y
    \frac{1}{N}\sum_{i=1}^N\left[y_i \varepsilon_i\right] = 0 \quad\Rightarrow\quad \sum_{i=1}^N\Bigl[y_i\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr] = 0
```

```{math}
    :label: EqGMM_LinReg_datacond_x1
    \frac{1}{N}\sum_{i=1}^N\left[x_{1,i} \varepsilon_i\right] = 0 \quad\Rightarrow\quad \sum_{i=1}^N\Bigl[x_{1,i}\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr] = 0
```

```{math}
    :label: EqGMM_LinReg_datacond_x2
    \frac{1}{N}\sum_{i=1}^N\left[x_{2,i} \varepsilon_i\right] = 0 \quad\Rightarrow\quad \sum_{i=1}^N\Bigl[x_{2,i}\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr] = 0
```

Think of the assumed zero correlations in equations {eq}`EqGMM_LinReg_momcond_y`, {eq}`EqGMM_LinReg_momcond_x1`, and {eq}`EqGMM_LinReg_momcond_x2` as data moments that are all equal to zero. And think of the empirical analogues of those moments as the left-hand-sides of equations {eq}`EqGMM_LinReg_datacond_y`, {eq}`EqGMM_LinReg_datacond_x1`, and {eq}`EqGMM_LinReg_datacond_x2` as the corresponding model moments. The exactly identified GMM approach to estimating the linear regression model in {eq}`EqGMM_LinReg_LinReg` is to choose the parameter vector $\theta=[\beta_0,\beta_1,\beta_2]$ to minimize the three moment error conditions,

```{math}
    :label: EqGMM_LinReg_exactprob
    \hat{\theta}_{lin,exact} = \theta:\quad \min_{\theta} e(x|\theta)^T\, W \, e(x|\theta) \\
    \text{where}\quad e(x|\theta)\equiv \begin{bmatrix}
      \sum_{i=1}^N\Bigl[y_i\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr] \\
      \sum_{i=1}^N\Bigl[x_{1,i}\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr] \\
      \sum_{i=1}^N\Bigl[x_{2,i}\left(y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i}\right)\Bigr]
    \end{bmatrix}
```

where $W$ is some $3\times 3$ weighting matrix.


(SecGMM_Exerc)=
## Exercises




(SecGMMfootnotes)=
## Footnotes

The footnotes from this chapter.

[^TruncNorm]: See Section {ref}`SecAppendixTruncNormal` of the Appendix for a description of the truncated normal distribution.
