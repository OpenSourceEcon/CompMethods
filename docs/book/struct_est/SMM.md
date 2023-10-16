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

(Chap_SMM)=
# Simulated Method of Moments Estimation

This chapter describes the simulated method of moments (SMM) estimation method. All data and images from this chapter can be found in the data directory ([./data/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/smm/)) and images directory ([./images/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/images/smm/)) for the GitHub repository for this online book.


(SecSMMestimator)=
## The SMM estimator

Simulated method of moments (SMM) is analogous to the generalized method of moments (GMM) estimator. SMM could really be thought of as a particular type of GMM estimator. The SMM estimator chooses a vector of model parameters $\theta$ to make simulated model moments match data moments. Seminal papers developing SMM are {cite}`McFadden:1989`, {cite}`LeeIngram:1991`, and {cite}`DuffieSingleton:1993`. Good textbook treatments of SMM are found in {cite}`AddaCooper:2003`, (pp. 87-100) and {cite}`DavidsonMacKinnon:2004`, (pp. 383-394).

Let the data be represented, in general, by $x$. This could have many variables, and it could be cross-sectional or time series. We define the estimation problem as one in which we want to model the data $x$ using some parameterized model $g(x|\theta)$ in which $\theta$ is a $K\times 1$ vector of parameters.

```{math}
    :label: EqSMM_ThetaVec
    \theta \equiv \left[\theta_1, \theta_2, ...\theta_K\right]^T
```

In the {ref}`Chap_MaxLikeli` chapter, we used data $x$ and model parameters $\theta$ to maximize the likelihood of drawing that data $x$ from the model given parameters $\theta$,

```{math}
    :label: EqSMM_MLestimator
    \hat{\theta}_{ML} = \theta:\quad \max_{\theta}\ln\mathcal{L} = \sum_{i=1}^N\ln\Bigl(f(x_i|\theta)\Bigr)
```

where $f(x_i|\theta)$ is the likelihood of seeing observation $x_i$ in the data $x$ given vector of parameters $\theta$.

In the {ref}`Chap_GMM` chapter, we used data $x$ and the $K\times 1$ vector of model parameters $\theta$ to minimize the distance between the vector of $R\geq K$ model moments $m(x|\theta)$ and data moments $m(x)$,

```{math}
    :label: EqSMM_GMMestimator
    \hat{\theta}_{GMM} = \theta:\quad \min_{\theta}||m(x|\theta) - m(x)||
```

where,

```{math}
    :label: EqSMM_ModMomFuncVecGen
    m(x|\theta) \equiv \left[m_1(x|\theta), m_2(x|\theta),...m_R(x|\theta)\right]^T
```

and,

```{math}
    :label: EqSMM_DataMomFuncVecGen
    m(x)\equiv \left[m_1(x), m_2(x), ...m_R(x)\right]^T
```

The following difficulties can arise with GMM making it not possible or very difficult.
* The model moment function $m(x|\theta)$ is not known analytically.
* The data moments you are trying to match come from another model (indirect inference, see {cite}`Smith:2020`).
* The model moments $m(x|\theta)$ are derived from *latent variables* that are not observed by the modeler. You only have moments, not the underlying data. See {cite}`LaroqueSalanie:1993`.
* The model moments $m(x|\theta)$ are derived from *censored variables* that are only partially observed by the modeler.
* The model moments $m(x|\theta)$ are just difficult to derive analytically. Examples include moments that include multiple integrals over nonlinear functions as in {cite}`McFadden:1989`.

SMM estimation is simply to simulate the model data $S$ times, and use the average values of the moments from the simulated data as the estimator for the model moments. Let $\tilde{x}\equiv\{\tilde{x}_1,\tilde{x}_2,...\tilde{x}_s,...\tilde{x}_S\}$ be the $S$ simulations of the model data. And let the maximization problem in {eq}`EqSMM_SMMestimator` be characterized by $R$ average moments across simulations, where $\hat{m}_r$ is the average value of the $r$th moment across the $S$ simulations where,

```{math}
    :label: EqSMM_AvgSimMoms_r
    \hat{m}_r\left(\tilde{x}|\theta\right) = \frac{1}{S}\sum_{s=1}^S m_r\left(\tilde{x}_s|\theta\right)
```

and

```{math}
    :label: EqSMM_AvgSimMoms_vec
    \hat{m}\left(\tilde{x}|\theta\right) = \left[m_1\left(\tilde{x}|\theta\right), m_2\left(\tilde{x}|\theta\right),...m_R\left(\tilde{x}|\theta\right)\right]^T
```

Once we have an estimate of the vector of $R$ average model moments $\hat{m}\left(\tilde{x}|\theta\right)$ from our $S$ simulations, SMM estimation is very similar to our presentation of GMM in {ref}`Chap_GMM`. The SMM approach of estimating the $K\times 1$ parameter vector $\hat{\theta}_{SMM}$ is to choose vector $\theta$ to minimize some distance measure of the $R$ data moments $m(x)$ from the $R$ simulated average model moments $\hat{m}(\tilde{x}|\theta)$.

```{math}
    :label: EqSMM_SMMestimator
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\: ||\hat{m}(\tilde{x}|\theta)-m(x)||
```

The distance measure $||\hat{m}(\tilde{x}|\theta)-m(x)||$ can be any kind of norm. But it is important to recognize that your estimates $\hat{\theta}_{SMM}$ will be dependent on what distance measure (norm) you choose. The most widely studied and used distance metric in GMM and SMM estimation is the $L^2$ norm or the sum of squared errors in moments.

Define the moment error vector $e(\tilde{x},x|\theta)$ as the $R\times 1$ vector of average moment error functions $e_r(\tilde{x},x|\theta)$ of the $r$th average moment error.

```{math}
    :label: EqSMM_MomError_vec
    e_(\tilde{x},x|\theta) \equiv \left[e_1(\tilde{x},x|\theta),e_2(\tilde{x},x|\theta),...e_R(\tilde{x},x|\theta)\right]^T
```

We can define the $r$th average moment error as the percent difference in the average simulated $r$th moment value $\hat{m}_r(\tilde{x}|\theta)$ from the $r$th data moment $m_r(x)$.

```{math}
    :label: EqSMM_MomError_r
    e_r(\tilde{x},x|\theta) \equiv \frac{\hat{m}_r(\tilde{x}|\theta)-m_r(x)}{m_r(x)} \quad\text{or}\quad \hat{m}_r(\tilde{x}|\theta)-m_r(x)
```

It is important that the error function $e_r(\tilde{x},x|\theta)$ be a percent deviation of the moments, although this will not work if the data moments are 0 or can be either positive or negative. This percent change transformation puts all the moments in the same units, which helps make sure that no moments receive unintended weighting simply due to its units. This ensures that the problem is scaled properly and will suffer from as little as possible ill conditioning.

In this case, the SMM estimator is the following,

```{math}
    :label: EqSMM_SMMestGen
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, W \, e(\tilde{x},x|\theta)
```

where $W$ is a $R\times R$ weighting matrix in the criterion function. For now, think of this weighting matrix as the identity matrix. But we will show in Section {ref}`SecSMM_WeightMatW` a more optimal weighting matrix. We call the quadratic form expression $e(\tilde{x},x|\theta)^T \, W \, e(\tilde{x},x|\theta)$ the *criterion function* because it is a strictly positive scalar that is the object of the minimization in the SMM problem statement. The $R\times R$ weighting matrix $W$ in the criterion function allows the econometrician to control how each moment is weighted in the minimization problem. For example, an $R\times R$ identity matrix for $W$ would give each moment equal weighting, and the criterion function would be a simply sum of squared percent deviations (errors). Other weighting strategies can be dictated by the nature of the problem or model.

One last item to emphasize with SMM, which we will highlight in the examples in this chapter, is that the errors that are drawn for the $S$ simulations of the model must be drawn only once so that the minimization problem for estimating $\hat{\theta}_{SMM}$ does not have the underlying sampling changing for each guess of a value of $\theta$. Put more simply, you want the random draws for all the simulations to be held constant so that the only thing changing in the minimization problem is the value of the vector of parameters $\theta$.


(SecSMM_WeightMatW)=
## The Weighting Matrix (W)

In the SMM criterion function in the problem statement above, some weighting matrices $W$ produce precise estimates while others produce poor estimates with large variances. We want to choose the optimal weighting matrix $W$ with the smallest possible asymptotic variance. This is an efficient or optimal SMM estimator. The optimal weighting matrix is the inverse variance covariance matrix of the moments at the optimal moments,

```{math}
    :label: EqSMM_estW_opt
    W^{opt} \equiv \Omega^{-1}(\tilde{x},x|\hat{\theta}_{SMM})
```

where $\Omega(\tilde{x},x|\theta)$ is the variance covariance matrix of the moment condition errors $e(\tilde{x},x|\theta)$. The intuition for using the inverse variance covariance matrix $\Omega^{-1}$ as the optimal weighting matrix is the following. You want to downweight moments that have a high variance, and you want to weight more heavily the moments that are generated more precisely.

Notice that this definition of the optimal weighting matrix is circular. $W^{opt}$ is a function of the SMM estimates $\hat{\theta}_{SMM}$, but the optimal weighting matrix is used in the estimation of $\hat{\theta}_{SMM}$. This means that one has to use some kind of iterative fixed point method to find the true optimal weighting matrix $W^{opt}$. Below are some examples of weighting matrices to use.


(SecSMM_W_I)=
### The identity matrix (W=I)

Many times, you can get away with just using the identity matrix as your weighting matrix $W = I$. This changes the criterion function to a simple sum of squared error functions such that each moment has the same weight.

```{math}
    :label: EqSMM_estW_I
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, e(\tilde{x},x|\theta)
```

If the problem is well conditioned and well identified, then your SMM estimates $\hat{\theta}_{SMM}$ will not be greatly affected by this simplest of weighting matrices.


(SecSMM_W_2step)=
### Two-step variance-covariance estimator of W

The most common method of estimating the optimal weighting matrix for SMM estimates is the two-step variance covariance estimator. The name "two-step" refers to the two steps used to get the weighting matrix.

The first step is to estimate the SMM parameter vector $\hat{\theta}_{1,SMM}$ using the simple identity matrix as the weighting matrix $W = I$.

```{math}
    :label: EqSMM_theta_2step_1
    \hat{\theta}_{1,SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, I \, e(\tilde{x},x|\theta)
```

Because we are simulating data, we can generate an estimator for the variance covariance matrix of the moment error vector $\hat{\Omega}$ using just the simulated data moments and the data moments. This $E(\tilde{x},x|\theta)$ matrix represents the contribution of the $s$th simulated moment to the $r$th moment error. Define $E(\tilde{x},x|\theta)$ as the $R\times S$ matrix of moment error functions from each simulation,

```{math}
    :label: EqSMM_estW_errmat_lev_1
    E(\tilde{x},x|\theta) =
      \begin{bmatrix}
        m_1(\tilde{x}_1|\theta) - m_1(x) & m_1(\tilde{x}_2|\theta) - m_1(x) & ... & m_1(\tilde{x}_S|\theta) - m_1(x) \\
        m_2(\tilde{x}_1|\theta) - m_2(x) & m_2(\tilde{x}_2|\theta) - m_2(x) & ... & m_2(\tilde{x}_S|\theta) - m_2(x) \\
        \vdots & \vdots & \ddots & \vdots \\
        m_R(\tilde{x}_1|\theta) - m_R(x) & m_R(\tilde{x}_2|\theta) - m_R(x) & ... & m_R(\tilde{x}_S|\theta) - m_R(x) \\
      \end{bmatrix}
```

where $m_r(x)$ is the $r$th data moment which is constant across each row, and $m_r(\tilde{x}_s|\theta)$ is the $r$th model moment from the $s$th simulation which are changing across each row. When the errors are percent deviations, the $E(\tilde{x},x|\theta)$ matrix is the following,

```{math}
    :label: EqSMM_estW_errmat_pct_1
    E(\tilde{x},x|\theta) =
      \begin{bmatrix}
        \frac{m_1(\tilde{x}_1|\theta) - m_1(x)}{m_1(x)} & \frac{m_1(\tilde{x}_2|\theta) - m_1(x)}{m_1(x)} & ... & \frac{m_1(\tilde{x}_S|\theta) - m_1(x)}{m_1(x)} \\
        \frac{m_2(\tilde{x}_1|\theta) - m_2(x)}{m_2(x)} & \frac{m_2(\tilde{x}_2|\theta) - m_2(x)}{m_2(x)} & ... & \frac{m_2(\tilde{x}_S|\theta) - m_2(x)}{m_2(x)} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{m_R(\tilde{x}_1|\theta) - m_R(x)}{m_R(x)} & \frac{m_R(\tilde{x}_2|\theta) - m_R(x)}{m_R(x)} & ... & \frac{m_R(\tilde{x}_S|\theta) - m_R(x)}{m_R(x)} \\
      \end{bmatrix}
```
where the denominator of the percentage deviation or baseline is the model moment that does not change. We use the $E(\tilde{x},x|\theta)$ data matrix and the Step 1 SMM estimate $e(x|\hat{\theta}_{1,SMM})$ to get a new estimate of the variance covariance matrix.

```{math}
    :label: EqSMM_2stepVarCov
    \hat{\Omega}_2 = \frac{1}{S}E(\tilde{x},x|\hat{\theta}_{1,SMM})\,E(\tilde{x},x|\hat{\theta}_{1,SMM})^T
``````

This is simply saying that the $(r,s)$-element of the estimator of the variance-covariance matrix of the moment vector is the following.

```{math}
    :label: EqSMM_2stepVarCov_rs
    \hat{\Omega}_{r,s} = \frac{1}{S}\sum_{i=1}^S\Bigl[m_r(\tilde{x}_i|\theta) - m_{r}(x)\Bigr]\Bigl[ m_s(\tilde{x}_i|\theta) - m_s(x)\Bigr]
``````

The optimal weighting matrix is the inverse of the two-step variance covariance matrix.

```{math}
    :label: EqSMM_estW_2step
    \hat{W}^{two-step} \equiv \hat{\Omega}_2^{-1}
```

Lastly, re-estimate the SMM estimator using the optimal two-step weighting matrix $\hat{W}^{2step}$.

```{math}
    :label: EqSMM_theta_2step_2
    \hat{\theta}_{2,SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, \hat{W}^{two-step} \, e(\tilde{x},x|\theta)
```

$\hat{\theta}_{2, SMM}$ is called the two-step SMM estimator.


(SecSMM_W_iter)=
### Iterated variance-covariance estimator of W

The truly optimal weighting matrix $W^{opt}$ is the iterated variance-covariance estimator of $W$. This procedure is to just repeat the process described in the two-step SMM estimator until the estimated weighting matrix no longer changes between iterations. Let $i$ index the $i$th iterated SMM estimator,

```{math}
    :label: EqSMM_theta_2step_i
    \hat{\theta}_{i, SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, \hat{W}_{i} \, e(\tilde{x},x|\theta)
```

and the $(i+1)$th estimate of the optimal weighting matrix is defined as the following.

```{math}
    :label: EqSMM_estW_istep
    \hat{W}_{i+1} \equiv \hat{\Omega}_{i+1}^{-1}\quad\text{where}\quad \hat{\Omega}_{i+1} = \frac{1}{S}E(\tilde{x},x|\hat{\theta}_{i,SMM})\,E(\tilde{x},x|\hat{\theta}_{i,SMM})^T
```

The iterated SMM estimator is the $\hat{\theta}_{i,SMM}$ such that $\hat{W}_{i+1}$ is very close to $\hat{W}_{i}$ for some distance metric (norm).

```{math}
    :label: EqSMM_theta_it
    \hat{\theta}_{it,SMM} = \hat{\theta}_{i,SMM}: \quad || \hat{W}_{i+1} - \hat{W}_{i} || < \varepsilon
```


(SecSMM_W_NW)=
### Newey-West consistent estimator of $\Omega$ and W

The Newey-West estimator of the optimal weighting matrix and variance covariance matrix is consistent in the presence of heteroskedasticity and autocorrelation in the data (See {cite}`NeweyWest:1987`). {cite}`AddaCooper:2003` (p. 82) have a nice exposition of how to compute the Newey-West weighting matrix $\hat{W}_{nw}$. The asymptotic representation of the optimal weighting matrix $\hat{W}^{opt}$ is the following:

```{math}
    :label: EqSMM_estW_WhatOpt
    \hat{W}^{opt} = \lim_{S\rightarrow\infty}\frac{1}{S}\sum_{i=1}^S \sum_{l=-\infty}^\infty E(\tilde{x}_i,x|\theta)E(\tilde{x}_{i-l},x|\theta)^T
```

The Newey-West consistend estimator of $\hat{W}^{opt}$ is:

```{math}
    :label: EqSMM_estW_NW
    \hat{W}_{nw} = \Gamma_{0,S} + \sum_{v=1}^q \left(1 - \left[\frac{v}{q+1}\right]\right)\left(\Gamma_{v,S} + \Gamma^T_{v,S}\right)
```

where

```{math}
    :label: EqSMM_estW_NWGamma
    \Gamma_{v,S} = \frac{1}{S}\sum_{i=v+1}^S E(\tilde{x}_i,x|\theta)E(\tilde{x}_{i-v},x|\theta)^T
```

Of course, for autocorrelation, the subscript $i$ can be changed to $t$.


(SecSMM_VarCovTheta)=
## Variance-Covariance Estimator of $\hat{\theta}$

Let the parameter vector $\theta$ have length $K$ such that $K$ parameters are being estimated. The estimated $K\times K$ variance-covariance matrix $\hat{\Sigma}$ of the estimated parameter vector $\hat{\theta}_{SMM}$ is different from the $R\times R$ variance-covariance matrix $\hat{\Omega}$ of the $R\times 1$ moment vector $e(\tilde{x},x|\theta)$ from the previous section.

Recall that each element of $e(\tilde{x},x|\theta)$ is an average moment error across all simulations. $\hat{\Omega}$ from the previous section is the $R\times R$ variance-covariance matrix of the $R$ moment errors used to identify the $K$ parameters $\theta$ to be estimated. The estimated variance-covariance matrix $\hat{\Sigma}$ of the estimated parameter vector is a $K\times K$ matrix. We say the model is *exactly identified* if $K = R$ (number of parameters $K$ equals number of moments $R$). We say the model is *overidentified* if $K<R$. We say the model is *not identified* or *underidentified* if $K>R$.

Similar to the inverse Hessian estimator of the variance-covariance matrix of the maximum likelihood estimator from the {ref}`Chap_MaxLikeli` chapter, the SMM variance-covariance matrix is related to the derivative of the criterion function with respect to each parameter. The intuition is that if the second derivative of the criterion function with respect to the parameters is large, there is a lot of curvature around the criterion minimizing estimate. In other words, the parameters of the model are precisely estimated. The inverse of the Hessian matrix will be small.

Define $R\times K$ matrix $d(\tilde{x},x|\theta)$ as the Jacobian matrix of derivatives of the $R\times 1$ error vector $e(\tilde{x},x|\theta)$ from {eq}`EqSMM_MomError_vec`.

```{math}
    :label: EqSMM_errvec_deriv
    \begin{equation}
      d(\tilde{x},x|\theta) \equiv
        \begin{bmatrix}
          \frac{\partial e_1(\tilde{x},x|\theta)}{\partial \theta_1} & \frac{\partial e_1(\tilde{x},x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_1(\tilde{x},x|\theta)}{\partial \theta_K} \\
          \frac{\partial e_2(\tilde{x},x|\theta)}{\partial \theta_1} & \frac{\partial e_2(\tilde{x},x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_2(\tilde{x},x|\theta)}{\partial \theta_K} \\
          \vdots & \vdots & \ddots & \vdots \\
          \frac{\partial e_R(\tilde{x},x|\theta)}{\partial \theta_1} & \frac{\partial e_R(\tilde{x},x|\theta)}{\partial \theta_2} & ... & \frac{\partial e_R(x|\theta)}{\partial \theta_K}
        \end{bmatrix}
    \end{equation}
```

The SMM estimates of the parameter vector $\hat{\theta}_{SMM}$ are assymptotically normal. If $\theta_0$ is the true value of the parameters, then the following holds,

```{math}
    :label: EqSMM_theta_plim
    \begin{equation}
      \text{plim}_{S\rightarrow\infty}\sqrt{S}\left(\hat{\theta}_{SMM} - \theta_0\right) \sim \text{N}\left(0, \left[d(\tilde{x},x|\theta)^T W d(\tilde{x},x|\theta)\right]^{-1}\right)
    \end{equation}
```

where $W$ is the optimal weighting matrix from the SMM criterion function. The SMM estimator for the variance-covariance matrix $\hat{\Sigma}_{SMM}$ of the parameter vector $\hat{\theta}_{SMM}$ is the following.

```{math}
    :label: EqSMM_SigmaHat
    \begin{equation}
      \hat{\Sigma}_{SMM} = \frac{1}{S}\left[d(\tilde{x},x|\theta)^T W d(\tilde{x},x|\theta)\right]^{-1}
    \end{equation}
```

In the examples below, we will use a finite difference method to compute numerical versions of the Jacobian matrix $d(\tilde{x},x|\theta)$. The following is a first-order forward finite difference numerical approximation of the first derivative of a function.

```{math}
    :label: EqSMM_finitediff_1
    f'(x_0) = \lim_{h\rightarrow 0} \frac{f(x_0 + h) - f(x_0)}{h}
```

The following is a centered second-order finite difference numerical approximation of the derivative of a function. (See [BYU ACME numerical differentiation lab](https://github.com/UC-MACSS/persp-model-econ_W19/blob/master/Notes/ACME_NumDiff.pdf) for more details.)

```{math}
    :label: EqSMM_finitediff_2
    f'(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h}
```


(SecSMM_CodeExmp)=
## Code Examples

In this section, we will use SMM to estimate parameters of the models from the {ref}`Chap_MaxLikeli` chapter and from the {ref}`Chap_GMM` chapter.

(SecSMM_CodeExmp_MacrTest)=
### Fitting a truncated normal to intermediate macroeconomics test scores

Let's revisit the problem from the MLE and GMM notebooks of fitting a truncated normal distribution to intermediate macroeconomics test scores. The data are in the text file [`Econ381totpts.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/smm/Econ381totpts.txt). Recall that these test scores are between 0 and 450. {numref}`Figure %s <FigSMM_EconScoreTruncNorm>` below shows a histogram of the data, as well as three truncated normal PDF's with different values for $\mu$ and $\sigma$. The black line is the maximum likelihood estimate of $\mu$ and $\sigma$ of the truncated normal pdf from the {ref}`Chap_MaxLikeli` chapter. The red, green, and black lines are just the PDF's of two "arbitrarily" chosen combinations of the truncated normal parameters $\mu$ and $\sigma$.[^TruncNorm]

```{code-cell} ipython3
:tags: ["hide-input", "remove-output"]

# Import the necessary libraries
import numpy as np
import scipy.stats as sts
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define function that generates values of a normal pdf
def trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cut_lb = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar lower bound value of distribution. Values below
             this value have zero probability
    cut_ub = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cut_ub == 'None' and cut_lb == 'None':
        prob_notcut = 1.0
    elif cut_ub == 'None' and cut_lb != 'None':
        prob_notcut = 1.0 - sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    elif cut_ub != 'None' and cut_lb == 'None':
        prob_notcut = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
    elif cut_ub != 'None' and cut_lb != 'None':
        prob_notcut = (sts.norm.cdf(cut_ub, loc=mu, scale=sigma) -
                       sts.norm.cdf(cut_lb, loc=mu, scale=sigma))

    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)

    return pdf_vals


# Download and save the data file Econ381totpts.txt
url = ('https://raw.githubusercontent.com/OpenSourceEcon/CompMethods/' +
       'main/data/smm/Econ381totpts.txt')
data_file = requests.get(url, allow_redirects=True)
open('../../../data/smm/Econ381totpts.txt', 'wb').write(data_file.content)

# Load the data as a NumPy array
data = np.loadtxt('../../../data/smm/Econ381totpts.txt')

num_bins = 30
count, bins, ignored = plt.hist(
    data, num_bins, density=True, edgecolor='k', label='data'
)
plt.title('Econ 381 scores: 2011-2012', fontsize=20)
plt.xlabel(r'Total points')
plt.ylabel(r'Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"

# Plot smooth line with distribution 1
dist_pts = np.linspace(0, 450, 500)
mu_1 = 300
sig_1 = 30
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_1, sig_1, 0, 450),
         linewidth=2, color='red', label=f"$\mu$={mu_1},$\sigma$={sig_1}")

# Plot smooth line with distribution 2
mu_2 = 400
sig_2 = 70
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_2, sig_2, 0, 450),
         linewidth=2, color='green', label=f"$\mu$={mu_2},$\sigma$={sig_2}")

# Plot smooth line with distribution 3
mu_3 = 558
sig_3 = 176
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_3, sig_3, 0, 450),
         linewidth=2, color='black', label=f"$\mu$={mu_3},$\sigma$={sig_3}")
plt.legend(loc='upper left')

plt.show()
```

```{figure} ../../../images/smm/Econ381scores_truncnorm.png
---
height: 500px
name: FigSMM_EconScoreTruncNorm
---
Macroeconomic midterm scores and three truncated normal distributions
```


(SecSMM_CodeExmp_MacrTest_2mI)=
#### Two moments, identity weighting matrix
Let's try estimating the parameters $\mu$ and $\sigma$ from the truncated normal distribution by SMM, assuming that we know the cutoff values for the distribution of scores $c_{lb}=0$ and $c_{ub}=450$. What moments should we use? Let's try the mean and variance of the data. These two statistics of the data are defined by:

$$ mean(scores_i) = \frac{1}{N}\sum_{i=1}^N scores_i $$

$$ var(scores_i) = \frac{1}{N-1}\sum_{i=1}^{N} \left(scores_i - mean(scores_i)\right)^2 $$

So the data moment vector $m(x)$ for SMM has two elements $R=2$ and is the following.

$$ m(scores_i) \equiv \begin{bmatrix} mean(scores_i) \\ var(scores_i) \end{bmatrix} $$

And the model moment vector $m(x|\theta)$ for SMM is the following.

$$ m(scores_i|\mu,\sigma) \equiv \begin{bmatrix} mean(scores_i|\mu,\sigma) \\ var(scores_i|\mu,\sigma) \end{bmatrix} $$

But let's assume that we need to simulate the data from the model (test scores) $S$ times in order to get the model moments. In this case, we don't need to simulate. But we will do so to show how SMM works.

```{code-cell} ipython3
:tags: ["remove-output"]

# Import packages and load the data
import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')

# Download and save the data file Econ381totpts.txt
url = ('https://raw.githubusercontent.com/OpenSourceEcon/CompMethods/' +
       'main/data/smm/Econ381totpts.txt')
data_file = requests.get(url, allow_redirects=True)
open('../../../data/smm/Econ381totpts.txt', 'wb').write(data_file.content)

# Load the data as a NumPy array
data = np.loadtxt('../../../data/smm/Econ381totpts.txt')
```

Let random variable $y\sim N(\mu,\sigma)$ be distributed normally with mean $\mu$ and standard deviation $\sigma$ with PDF given by $\phi(y|\mu,\sigma)$ and CDF given by $\Phi(y|\mu,\sigma)$. The truncated normal distribution of random variable $x\in(a,b)$ based on $y$ but with cutoff values of $a\geq -\infty$ as a lower bound and $a < b\leq\infty$ as an upper bound has the following probability density function.

$$ f(x|\mu,\sigma,a,b) = \begin{cases} 0 \quad\text{if}\quad x\leq a \\ \frac{\phi(x|\mu,\sigma)}{\Phi(b|\mu,\sigma) - \Phi(a|\mu,\sigma)}\quad\text{if}\quad a < x < b \\ 0 \quad\text{if}\quad x\geq b \end{cases} $$

The CDF of the truncated normal can be shown to be the following:

$$ F(x|\mu,\sigma,a,b) = \begin{cases} 0 \quad\text{if}\quad x\leq a \\ \frac{\Phi(x|\mu,\sigma) - \Phi(a|\mu,\sigma)}{\Phi(b|\mu,\sigma) - \Phi(a|\mu,\sigma)}\quad\text{if}\quad a < x < b \\ 0 \quad\text{if}\quad x\geq b \end{cases} $$

The inverse CDF of the truncated normal takes a value $p$ between 0 and 1 and solves for the value of $x$ for which $p=F(x|\mu,\sigma,a,b)$. The expression for the inverse CDF of the truncated normal is the following:

$$ x = \Phi^{-1}(z|\mu,\sigma) \quad\text{where}\quad z = p\Bigl[\Phi(b|\mu,\sigma) - \Phi(a|\mu,\sigma)\Bigr] + \Phi(a|\mu,\sigma) $$

Note that $z$ is just a transformation of $p$ such that $z\sim U\Bigl(\Phi^{-1}(a|\mu,\sigma), \Phi^{-1}(b|\mu,\sigma)\Bigr)$.

The following code for `trunc_norm_pdf()` is a function that returns the probability distribution function value of random variable value $x$ given parameters $\mu$, $\sigma$, $c_{lb}$, $c_{ub}$.

```{code-cell} ipython3
:tags: ["remove-output"]

def trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cut_lb = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar lower bound value of distribution. Values below
             this value have zero probability
    cut_ub = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cut_ub == 'None' and cut_lb == 'None':
        prob_notcut = 1.0
    elif cut_ub == 'None' and cut_lb != 'None':
        prob_notcut = 1.0 - sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    elif cut_ub != 'None' and cut_lb == 'None':
        prob_notcut = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
    elif cut_ub != 'None' and cut_lb != 'None':
        prob_notcut = (sts.norm.cdf(cut_ub, loc=mu, scale=sigma) -
                       sts.norm.cdf(cut_lb, loc=mu, scale=sigma))

    pdf_vals = (
        (1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (xvals - mu)**2 / (2 * sigma**2))) /
        prob_notcut
    )

    return pdf_vals
```

The following code `trunc_norm_draws` is a function that draws $S$ simulations of $N$ observations of the random variable $x_{n,s}$ that is distributed truncated normal. This function takes as an input an $N\times S$ matrix of uniform distributed values $u_{n,s}\sim U(0,1)$.

```{code-cell} ipython3
:tags: ["remove-output"]

def trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a truncated normal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()

    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                  cutoff of truncated normal distribution
    cut_lb_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                  cutoff of truncated normal distribution
    unif2_vals  = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  rescaled uniform derived from original.
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma) and cutoffs
                  (cut_lb, cut_ub)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''
    # No cutoffs: truncated normal = normal
    if (cut_lb == None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb != None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb == None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb != None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)

    unif2_vals = unif_vals * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    tnorm_draws = sts.norm.ppf(unif2_vals, loc=mu, scale=sigma)

    return tnorm_draws
```

What would one simulation of 161 test scores look like from a truncated normal with mean $\mu=300$, $\sigma=30$?

```{code-cell} ipython3
:tags: []

mu_1 = 300.0
sig_1 = 30.0
cut_lb_1 = 0.0
cut_ub_1 = 450.0
np.random.seed(seed=1975)  # Set seed so the simulation values are always the same
unif_vals_1 = sts.uniform.rvs(0, 1, size=161)
draws_1 = trunc_norm_draws(unif_vals_1, mu_1, sig_1, cut_lb_1, cut_ub_1)
print('Mean of simulated score =', draws_1.mean())
print('Variance of simulated scores =', draws_1.var())
print('Standard deviation of simulated scores =', draws_1.std())
```

```{code-cell} ipython3
:tags: ["remove-output"]

# Plot data histogram vs. simulated data histogram
count_d, bins_d, ignored_d = \
    plt.hist(data, 30, density=True, color='b', edgecolor='black',
             linewidth=0.8, label='Data')
count_m, bins_m, ignored_m = \
    plt.hist(draws_1, 30, density=True, color='r', edgecolor='black',
             linewidth=0.8, alpha=0.5, label='Simulated data')
xvals = np.linspace(0, 450, 500)
plt.plot(xvals, trunc_norm_pdf(xvals, mu_1, sig_1, cut_lb_1, cut_ub_1),
         linewidth=2, color='k', label='PDF, simulated data')
plt.title('Econ 381 scores: 2011-2012', fontsize=20)
plt.xlabel('Total points')
plt.ylabel('Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"
plt.legend(loc='upper left')

plt.show()
```

```{figure} ../../../images/smm/Econ381scores_sim1.png
---
height: 500px
name: FigSMM_EconScoreSim1
---
Histograms of one simulation of 161 Econ 381 test scores (2011-2012) from arbitrary truncated normal distribution compared to data
```

From that simulation, we can calculate moments from the simulated data just like we did from the actual data. The following function `data_moments2()` computes the mean and the variance of the simulated data $x$, where $x$ is an $N\times S$ matrix of $S$ simulations of $N$ observations each.

```{code-cell} ipython3
:tags: []

def data_moments2(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix or (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        var_data = xvals.var()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        var_data = xvals.var(axis=0)

    return mean_data, var_data
```

```{code-cell} ipython3
:tags: []

mean_data, var_data = data_moments2(data)
print('Data mean =', mean_data)
print('Data variance =', var_data)
mean_sim, var_sim = data_moments2(draws_1)
print('Sim. mean =', mean_sim)
print('Sim. variance =', var_sim)
```

We can also simulate many $(S)$ data sets of test scores, each with $N=161$ test scores. The estimate of the model moments will be the average of the simulated data moments across the simulations.

```{code-cell} ipython3
:tags: []

N = 161
S = 100
mu_2 = 300.0
sig_2 = 30.0
cut_lb = 0.0
cut_ub = 450.0
np.random.seed(25)  # Set the random number seed to get same answers every time
unif_vals_2 = sts.uniform.rvs(0, 1, size=(N, S))
draws_2 = trunc_norm_draws(unif_vals_2, mu_2, sig_2,
                           cut_lb, cut_ub)

mean_sim, var_sim = data_moments2(draws_2)
print("Mean test score in each simulation:")
print(mean_sim)
print("")
print("Variance of test scores in each simulation:")
print(var_sim)
mean_mod = mean_sim.mean()
var_mod = var_sim.mean()
print("")
print('Estimated model mean (avg. of means) =', mean_mod)
print('Estimated model variance (avg. of variances) =', var_mod)
```

Our SMM model moments $\hat{m}(\tilde{scores}_i|\mu,\sigma)$ are an estimate of the true models moments that we got in the GMM case by integrating using the PDF of the truncated normal distribution. Our SMM moments we got by simulating the data $S$ times and taking the average of the simulated data moments across the simulations as our estimator of the model moments.

Define the error vector as the vector of percent deviations of the model moments from the data moments.

$$ e(\tilde{scores}_i,scores_i|\mu,\sigma) \equiv \frac{\hat{m}(\tilde{scores}_i|\mu,\sigma) - m(scores_i)}{m(scores_i)} $$

The SMM estimator for this moment vector is the following.

$$ (\hat{\mu}_{SMM},\hat{\sigma}_{SMM}) = (\mu,\sigma):\quad \min_{\mu,\sigma} e(\tilde{scores}_i,scores_i|\mu,\sigma)^T \, W \, e(\tilde{scores}_i,scores_i|\mu,\sigma) $$

Now let's define a criterion function that takes as inputs the parameters and the estimator for the weighting matrix $\hat{W}$.

```{code-cell} ipython3
:tags: []

def err_vec2(data_vals, unif_vals, mu, sigma, cut_lb, cut_ub, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    unif_vals = (N, S) matrix, S simulations of N observations from
                uniform distribution U(0,1)
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_draws()
        data_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    mean_data, var_data = data_moments2(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments2(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 6 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat, simple)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    simple    = Boolean, =True if error vec is simple difference,
                =False if error vec is percent difference

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec2()

    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, cut_lb, cut_ub, W_hat, simple = args
    err = err_vec2(xvals, unif_vals, mu, sigma, cut_lb, cut_ub,
                   simple)
    crit_val = err.T @ W_hat @ err

    return crit_val
```

```{code-cell} ipython3
:tags: []

mu_test = 400
sig_test = 70
cut_lb = 0.0
cut_ub = 450.0
sim_vals = trunc_norm_draws(unif_vals_2, mu_test, sig_test, cut_lb, cut_ub)
mean_sim, var_sim = data_moments2(sim_vals)
mean_mod = mean_sim.mean()
var_mod = var_sim.mean()
err_vec2(data, unif_vals_2, mu_test, sig_test, cut_lb, cut_ub, simple=False)
crit_test = criterion(np.array([mu_test, sig_test]), data, unif_vals_2,
                      0.0, 450.0, np.eye(2), False)
print("Average of mean test scores across simulations is:", mean_mod)
print("")
print("Average variance of test scores across simulations is:", var_mod)
print("")
print("Criterion function value is:", crit_test[0][0])
```

Now we can perform the SMM estimation using SciPy's minimize function to choose the values of $\mu$ and $\sigma$ of the truncated normal distribution that best fit the data by minimizing the crietrion function. Let's start with the identity matrix as our estimate for the optimal weighting matrix $W = I$.

```{code-cell} ipython3
:tags: []

mu_init_1 = 300
sig_init_1 = 30
params_init_1 = np.array([mu_init_1, sig_init_1])
W_hat1_1 = np.eye(2)
smm_args1_1 = (data, unif_vals_2, cut_lb, cut_ub, W_hat1_1, False)
results1_1 = opt.minimize(criterion, params_init_1, args=(smm_args1_1),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM1_1, sig_SMM1_1 = results1_1.x
print('mu_SMM1_1=', mu_SMM1_1, ' sig_SMM1_1=', sig_SMM1_1)
```

```{code-cell} ipython3
:tags: []

mean_data, var_data = data_moments2(data)
print('Data mean of scores =', mean_data, ', Data variance of scores =', var_data)
sim_vals_1 = trunc_norm_draws(unif_vals_2, mu_SMM1_1, sig_SMM1_1, cut_lb, cut_ub)
mean_sim_1, var_sim_1 = data_moments2(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
var_model_1 = var_sim_1.mean()
err_1 = err_vec2(data, unif_vals_2, mu_SMM1_1, sig_SMM1_1, cut_lb, cut_ub,
                 False).reshape(2,)
print("")
print('Model mean 1 =', mean_model_1, ', Model variance 1 =', var_model_1)
print("")
print('Error vector 1 =', err_1)
print("")
print("Results from scipy.opmtimize.minimize:")
print(results1_1)
```

Let's plot the PDF implied by these SMM estimates $(\hat{\mu}_{SMM},\hat{\sigma}_{SMM})=(612.337, 197.264)$ against the histogram of the data in {numref}`Figure %s <FigSMM_Econ381_SMM1>` below.

```{code-cell} ipython3
:tags: ["remove-output"]

# Plot the histogram of the data
count, bins, ignored = plt.hist(data, 30, density=True,
                                edgecolor='black', linewidth=1.2, label='data')
plt.title('Econ 381 scores: 2011-2012', fontsize=20)
plt.xlabel('Total points')
plt.ylabel('Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 450, 500)
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1, 0.0, 450.0),
         linewidth=2, color='k', label='PDF: ($\hat{\mu}_{SMM1}$,$\hat{\sigma}_{SMM1}$)=(612.34, 197.26)')
plt.legend(loc='upper left')

plt.show()
```

```{figure} ../../../images/smm/Econ381scores_smm1.png
---
height: 500px
name: FigSMM_Econ381_SMM1
---
SMM-estimated PDF function and data histogram, 2 moments, identity weighting matrix, Econ 381 scores (2011-2012)
```

That looks just like the maximum likelihood estimate from the {ref}`Chap_MaxLikeli` chapter. {numref}`Figure %s <FigSMM_Econ381_crit1>` below shows what the minimizer is doing. The figure shows the criterion function surface for different of $\mu$ and $\sigma$ in the truncated normal distribution. The minimizer is searching for the parameter values that give the lowest criterion function value.

```{code-cell} ipython3
:tags: ["remove-output"]

mu_vals = np.linspace(60, 700, 90)
sig_vals = np.linspace(20, 250, 100)
crit_vals = np.zeros((90, 100))
crit_args = (data, unif_vals_2, cut_lb, cut_ub, W_hat1_1, False)
for mu_ind in range(90):
    for sig_ind in range(100):
        crit_params = np.array([mu_vals[mu_ind], sig_vals[sig_ind]])
        crit_vals[mu_ind, sig_ind] = criterion(crit_params, *crit_args)[0][0]

mu_mesh, sig_mesh = np.meshgrid(mu_vals, sig_vals)

crit_SMM1_1 = criterion(np.array([mu_SMM1_1, sig_SMM1_1]), *crit_args)[0][0]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(mu_mesh.T, sig_mesh.T, crit_vals, rstride=8,
                cstride=1, cmap=cmap1, alpha=0.9)
ax.scatter(mu_SMM1_1, sig_SMM1_1, crit_SMM1_1, color='red', marker='o',
           s=18, label='SMM1 estimate')
ax.view_init(elev=12, azim=30, roll=0)
ax.set_title('Criterion function for values of mu and sigma')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel(r'Crit. func.')
plt.tight_layout()

plt.show()
```

```{figure} ../../../images/smm/Econ381_crit1.png
---
height: 500px
name: FigSMM_Econ381_crit1
---
Criterion function surface for values of $\mu$ and $\sigma$ for SMM estimation of truncated normal with two moments and identity weighting matrix (SMM estimate shown as red dot)
```

Let's compute the SMM estimator for the variance-covariance matrix $\hat{\Sigma}_{SMM}$ of our SMM estimates $\hat{\theta}_{SMM}$ using the equation in Section {ref}`SecSMM_VarCovTheta` based on the Jacobian $d(\tilde{x},x|\hat{\theta}_{SMM})$ of the moment error vector $e(\tilde{x},x|\hat{\theta}_{SMM})$ from the criterion function at the estimated (optimal) parameter values $\hat{\theta}_{SMM}$. We first write a function that computes the Jacobian matrix $d(x|\hat{\theta}_{SMM})$, which has shape $2\times 2$ in this case with two moments $R=2$.

```{code-cell} ipython3
:tags: []

def Jac_err2(data_vals, unif_vals, mu, sigma, cut_lb, cut_ub, simple=False):
    '''
    This function computes the Jacobian matrix of partial derivatives of the
    R x 1 moment error vector e(x|theta) with respect to the K parameters
    theta_i in the K x 1 parameter vector theta. The resulting matrix is R x K
    Jacobian.
    '''
    Jac_err = np.zeros((2, 2))
    h_mu = 1e-4 * mu
    h_sig = 1e-4 * sigma
    Jac_err[:, 0] = (
        (err_vec2(xvals, unif_vals, mu + h_mu, sigma, cut_lb, cut_ub, simple) -
         err_vec2(xvals, unif_vals, mu - h_mu, sigma, cut_lb, cut_ub, simple)) /
        (2 * h_mu)
    ).flatten()
    Jac_err[:, 1] = (
        (err_vec2(xvals, unif_vals, mu, sigma + h_sig, cut_lb, cut_ub, simple) -
         err_vec2(xvals, unif_vals, mu, sigma - h_sig, cut_lb, cut_ub, simple)) /
        (2 * h_sig)
    ).flatten()

    return Jac_err
```

```{code-cell} ipython3
:tags: []

S = unif_vals_2.shape[1]
d_err2 = Jac_err2(data, unif_vals_2, mu_SMM1_1, sig_SMM1_1, 0.0, 450.0, False)
print("Jacobian matrix of derivatives of moment error functions is:")
print(d_err2)
print("")
print("Weighting matrix W is:")
print(W_hat1_1)
SigHat2 = (1 / S) * lin.inv(d_err2.T @ W_hat1_1 @ d_err2)
print("")
print("Variance-covariance matrix of estimated parameter vector is:")
print(SigHat2)
print("")
print('Std. err. mu_hat=', np.sqrt(SigHat2[0, 0]))
print('Std. err. sig_hat=', np.sqrt(SigHat2[1, 1]))
```

This SMM estimation methodology of estimating $\mu$ and $\sigma$ from the truncated normal distribution to fit the distribution of Econ 381 test scores using two moments from the data and using the identity matrix as the optimal weighting matrix is not very precise. The standard errors for the estimates of $\hat{mu}$ and $\hat{sigma}$ are bigger than their values.

In the next section, we see if we can get more accurate estimates (lower criterion function values) of $\hat{mu}$ and $\hat{sigma}$ with more precise standard errors by using the two-step optimal weighting matrix described in Section {ref}`SecSMM_W_2step`.


(SecSMM_CodeExmp_MacrTest_2m2st)=
#### Two moments, two-step optimal weighting matrix
Similar to the maximum likelihood estimation problem in Chapter {ref}`Chap_MaxLikeli`, it looks like the minimum value of the criterion function shown in {numref}`Figure %s <FigSMM_Econ381_crit1>` is roughly equal for a specific portion increase of $\mu$ and $\sigma$ together. That is, the estimation problem with these two moments probably has a correspondence of values of $\mu$ and $\sigma$ that give roughly the same minimum criterion function value. This issue has two possible solutions.

1. Maybe we need the two-step variance covariance estimator to calculate a "more" optimal weighting matrix $W$.
2. Maybe our two moments aren't very good moments for fitting the data.

Let's first try the two-step weighting matrix.

```{code-cell} ipython3
:tags: []

def get_Err_mat2(pts, unif_vals, mu, sigma, cut_lb, cut_ub, simple=False):
    '''
    --------------------------------------------------------------------
    This function computes the R x S matrix of errors from each
    simulated moment for each moment error. In this function, we have
    hard coded R = 2.
    --------------------------------------------------------------------
    INPUTS:
    xvals     = (N,) vector, test scores data
    unif_vals = (N, S) matrix, uniform random variables that generate
                the N observations of simulated data for S simulations
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    cut_lb    = scalar or string, ='None' if no cutoff is given,
                otherwise is scalar lower bound value of distribution.
                Values below this value have zero probability
    cut_ub    = scalar or string, ='None' if no cutoff is given,
                otherwise is scalar upper bound value of distribution.
                Values above this value have zero probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        model_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    R          = integer = 2, hard coded number of moments
    S          = integer >= R, number of simulated datasets
    Err_mat    = (R, S) matrix, error by moment and simulated data
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Err_mat
    --------------------------------------------------------------------
    '''
    R = 2
    S = unif_vals.shape[1]
    Err_mat = np.zeros((R, S))
    mean_data, var_data = data_moments2(pts)
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    mean_model, var_model = data_moments2(sim_vals)
    if simple:
        Err_mat[0, :] = mean_model - mean_data
        Err_mat[1, :] = var_model - var_data
    else:
        Err_mat[0, :] = (mean_model - mean_data) / mean_data
        Err_mat[1, :] = (var_model - var_data) / var_data

    return Err_mat
```

```{code-cell} ipython3
:tags: []

Err_mat2 = get_Err_mat2(data, unif_vals_2, mu_SMM1_1, sig_SMM1_1, 0.0, 450.0, False)
VCV2 = (1 / unif_vals_2.shape[1]) * (Err_mat2 @ Err_mat2.T)
print("2nd stage est. of var-cov matrix of moment error vec across sims:")
print(VCV2)
W_hat2_1 = lin.inv(VCV2)
print("")
print("2nd state est. of optimal weighting matrix:")
print(W_hat2_1)
```

```{code-cell} ipython3
:tags: []

params_init2_1 = np.array([mu_SMM1_1, sig_SMM1_1])
smm_args2_1 = (data, unif_vals_2, cut_lb, cut_ub, W_hat2_1, False)
results2_1 = opt.minimize(criterion, params_init2_1, args=(smm_args2_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2_1, sig_SMM2_1 = results2_1.x
print('mu_SMM2_1=', mu_SMM2_1, ' sig_SMM2_1=', sig_SMM2_1)
```

Look at how much smaller (more efficient) the estimated standard errors are in this case with the two-step optimal weighting matrix $\hat{W}_{2step}$.

```{code-cell} ipython3
:tags: []

d_err2_2 = Jac_err2(data, unif_vals_2, mu_SMM2_1, sig_SMM2_1, 0.0, 450.0, False)
print("Jacobian matrix of derivatives of moment error functions is:")
print(d_err2_2)
print("")
print("Weighting matrix W is:")
print(W_hat2_1)
SigHat2_2 = (1 / S) * lin.inv(d_err2_2.T @ W_hat2_1 @ d_err2_2)
print("")
print("Variance-covariance matrix of estimated parameter vector is:")
print(SigHat2_2)
print("")
print('Std. err. mu_hat=', np.sqrt(SigHat2_2[0, 0]))
print('Std. err. sig_hat=', np.sqrt(SigHat2_2[1, 1]))
```


(SecSMM_CodeExmp_MacrTest_4mI)=
#### Four moments, identity matrix weighting matrix

Using a better weighting matrix didn't improve our estimates or fit very much---the estimates of $\hat{mu}$ and $\hat{\sigma}$ and the corresponding minimum criterion function value. But it did improve our standard errors. But even with the optimal weighting matrix, our standard errors still look pretty big. This might mean that we did not choose good moments for fitting the data. Let's try some different moments. How about four moments to match.

1. The percent of observations greater than 430 (between 430 and 450)
2. The percent of observations between 320 and 430
3. The percent of observations between 220 and 320
4. The percent of observations less than 220 (between 0 and 220)

This means we are using four moments $R=4$ to identify two paramters $\mu$ and $\sigma$ ($K=2$). This problem is now overidentified ($R>K$). This is often a desired approach for SMM estimation.


(SecSMM_CodeExmp_MacrTest_4m2st)=
#### Four moments, two-step optimal weighting matrix



(SecSMM_CodeExmp_BM72)=
### Brock and Mirman (1972) estimation by SMM
In {numref}`ExercStructEst_SMM_BM72`, you will estimate four parameters in the {cite}`BrockMirman:1972` macroeconomic model by simulating the model to get six moments.


(SecSMM_Ident)=
## Identification

An issue that we saw in the examples from the previous section is that there is some science as well as some art in choosing moments to identify the parameters in an SMM estimation as well as in GMM. Suppose the parameter vector $\theta$ has $K$ elements, or rather, $K$ parameters to be estimated. In order to estimate $\theta$ by GMM, you must have at least as many moments as parameters to estimate $R\geq K$. If you have exactly as many moments as parameters to be estimated $R=K$, the model is said to be *exactly identified*. If you have more moments than parameters to be estimated $R>K$, the model is said to be *overidentified*. If you have fewer moments than parameters to be estimated $R<K$, the model is said to be *underidentified*. There are good reasons to overidentify $R>K$ the model in SMM estimation as we saw in the previous example. The main reason is that not all moments are orthogonal. That is, some moments convey roughly the same information about the data and, therefore, do not separately identify any extra parameters. So a good SMM model often is overidentified $R>K$.

One last point about MM regards moment selection and verification of results. The real world has an infinite supply of potential moments that describe some part of the data. Choosing moments to estimate parameters by SMM requires understanding of the model, intuition about its connections to the real world, and artistry. A good SMM estimation will include moments that have some relation to or story about their connection to particular parameters of the model to be estimated. In addition, a good verification of a SMM estimation is to take some moment from the data that was not used in the estimation and see how well the corresponding moment from the estimated model matches that *outside moment*.


(SecSMM_IndirInf)=
## Indirect inference

Indirect inference is a particular application of SMM with some specific characteristics. As moments to match it uses parameters of an auxiliary model that can be estimated both on the real-world data and on the simulated data. {cite}`Smith:2020` gives a great summary of the topic with some examples. See also {cite}`GourierouxMonfort:1996` (ch. 4) for a textbook treatment of the topic.


(SecSMM_IndirInf_SMMprob)=
### Restatement of the general SMM estimation problem

Define a model or data generating process (DGP) as a system of equations,

$$ G(x_t,z_t|\theta)=0 $$

which are functions of a vector of endogenous variables $x_t$, exogenous variables $z_t$, and parameters $\theta$. In the general simulated method of moments (SMM) estimation approach, one would choose data moments $m(x_t,z_t)$ that are just statistics of the data and model moments $\hat{m}(\tilde{x}_t,\tilde{z}_t|\theta)$ that are averages of the same data moments calculated on simulated samples of the data. The SMM estimator is to choose the parameter vector $\hat{\theta}_{SMM}$ to minimize some distance of the model moments from the data moments.


$$ \hat{\theta}_{SMM}=\theta:\quad \min_{\theta} ||\hat{m}(\tilde{x}_t,\tilde{z}_t|\theta) - m(x_t,z_t)|| $$


(SecSMM_IndirInf_IndInfprob)=
### Indirect inference estimation problem

Indirect inference is to change the model moments from being stastics that are calculated directly from the simulated data to being statistics that are calculated indirectly from the simulated data. These indirect inference model moments are parameters from an auxiliary model.

Let an auxiliary model be defined as $H(x_t,z_t|\phi)=0$. The parameters of the auxiliary model $\phi$ will be the moments we use to identify the model parameters $\theta$. Suppose that the model parameter vector $\theta$ has $K$ elements. Then the auxiliary model parameter vector $\phi$ must have $R$ elements such that $R\geq K$. This is the typical identification restriction that the number of model moments must be at least as many as the number of model parameters being estimated.

When the auxiliary model is run on real-world data $H(x_t,z_t|\phi)=0$, the resulting values of the auxiliary model parameters are the data moments $\hat{\phi}(x_t,z_t)$. Note that these data moments $\hat{\phi}$ have a hat on them to represent that these moments are usually estimated in some way. When the auxiliary model is run on the $s$th simulation of the data given model parameters $H(\tilde{x}_{s,t},\tilde{z}_{s,t}|\phi)=0$, the auxiliary model parameters are the $s$th estimate of the model moments $\hat{\phi}_s(\tilde{x}_{s,t},\tilde{z}_{s,t}|\theta)$. The model moments are then the average of these auxiliary model parameter estimates across the simulations.

$$ \hat{\phi}(\tilde{x}_{t},\tilde{z}_{t}|\theta) = \frac{1}{S}\sum_{s=1}^S \hat{\phi}_s(\tilde{x}_{s,t},\tilde{z}_{s,t}|\theta) $$

The indirect inference estimation method is simply to choose a model parameter vector $\theta$ that minimizes some distance metric between the model moments $\hat{\phi}(\tilde{x}_{t},\tilde{z}_{t}|\theta)$ and the data moments $\hat{\phi}(x_t,z_t)$.

$$ \hat{\theta}_{SMM}=\theta:\quad \min_{\theta} ||\hat{\phi}(\tilde{x}_{t},\tilde{z}_{t}|\theta) - \hat{\phi}(x_t,z_t)|| $$

In most examples of indirect, the data moments and model moments are some regression of endogenous variables on exogenous variables. In the univariate case, it is usually linear regression. In the multivariate case, it is usually a vector autoregression (VAR). But most examples are reduced form parameter estimation exercises. Other examples are probit, logit, and two-stage IV regressions. The key is that these statistics be computationally tractable and have convenient or accurate data availability.


(SecSMM_IndirInf_HypothTest)=
### Hypothesis testing with indirect inference

* Wald test
* likelihood ratio test


(SecSMM_Exerc)=
## Exercises

```{exercise-start} Estimating the Brock and Mirman (1972) model by SMM
:label: ExercStructEst_SMM_BM72
:class: green
```
You can observe time series data in an economy for the following variables: $(c_t, k_t, w_t, r_t, y_t)$. The data can be loaded from the file [`NewMacroSeries.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/smm/NewMacroSeries.txt) in the online book repository data folder `data/smm/`. This file is a comma separated text file with no labels. The variables are ordered as $(c_t, k_t, w_t, r_t, y_t)$. These data have 100 periods, which are quarterly (25 years). Suppose you think that the data are generated by a process similar to the {cite}`BrockMirman:1972` paper. A simplified set of characterizing equations of the Brock and Mirman model are the following six equations.
```{math}
    :label: EqSMM_BM72_eul
    (c_t)^{-1} - \beta E\left[r_{t+1}(c_{t+1})^{-1}\right] = 0
```
```{math}
    :label: EqSMM_BM72_bc
    c_t + k_{t+1} - w_t - r_t k_t = 0
```
```{math}
    :label: EqSMM_BM72_focl
    w_t - (1-\alpha)e^{z_t}(k_t)^\alpha = 0
```
```{math}
    :label: EqSMM_BM72_fock
    r_t - \alpha e^{z_t}(k_t)^{\alpha-1} = 0
```
```{math}
    :label: EqSMM_BM72_zt
    z_t = \rho z_{t-1} + (1-\rho)\mu + \varepsilon_t \quad\text{where}\quad \varepsilon_t\sim N(0,\sigma^2)
```
```{math}
    :label: EqSMM_BM72_prod
    y_t = e^{z_t}(k_t)^\alpha
```
The variable $c_t$ is aggregate consumption in period $t$, $k_{t+1}$ is total household savings and investment in period $t$ for which they receive a return in the next period $t+1$ (this model assumes full depreciation of capital). The wage per unit of labor in period $t$ is $w_t$, and the interest rate or rate of return on investment is
$r_t$. Total factor productivity is $z_t$, which follows an AR(1) process given in {eq}`EqSMM_BM72_zt`. GDP is $y_t$. The rest of the symbols in the equations are parameters that must be estimated or calibrated $(\alpha, \beta, \rho, \mu, \sigma)$. The constraints on these parameters are the following.
\begin{equation*}
  \alpha,\beta \in (0,1),\quad \mu,\sigma > 0, \quad\rho\in(-1,1)
\end{equation*}
Assume that the first observation in the data file variables is $t=1$. Let $k_1$ be the first observation in the data fil for the variable $k_t$. One nice property of the {cite}`BrockMirman:1972` model is that the household decision has a known analytical solution in which the optimal savings decision $k_{t+1}$ is a function of the productivity shock today $z_t$ and the amount of capital today $k_t$.
```{math}
    :label: EqSMM_BM72_pf
    k_{t+1} = \alpha\beta e^{z_t}(k_t)^\alpha
```
With this solution {eq}`EqSMM_BM72_pf` and equations {eq}`EqSMM_BM72_bc` through {eq}`EqSMM_BM72_zt`, it is straightforward to simulate the data of the {cite}`BrockMirman:1972` model given parameters $(\alpha, \beta, \rho, \mu, \sigma)$.

First, assume that $z_0=\mu$ and that $k_1=\text{mean}(k_t)$ from the data. These are initial values that will not change across simulations. Also assume that $\beta=0.99$.

Next, draw a matrix of $S=1,000$ simulations (columns) of $T=100$ (rows) from a uniform distribution $u_{s,t}\sim U(0,1)$. These draws will not change across this SMM estimation procedure.

For each guess of the parameter vector $(\alpha,\rho,\mu,\sigma)$ given $\beta=0.99$, you can use $u_{s,t}$ to generate normally distributed errors $\varepsilon_{s,t}\sim N(0,\sigma^2)$ using the inverse cdf of the normal distribution, where $s$ is the index of the simulation number (columns).

With $\varepsilon_{s,t}$, $\rho$, $\mu$, and $z_0=\mu$, you can use {eq}`EqSMM_BM72_zt` to generate the simulationed values for $z_{s,t}$.

With $\alpha$, $\beta=0.99$, $z_{s,t}$, and $k_1$, you can use {eq}`EqSMM_BM72_pf` to generate simulated values for $k_{t+1}$.

With $\alpha$, $z_{s,t}$, and $k_{s,t}$, you can use {eq}`EqSMM_BM72_focl` and {eq}`EqSMM_BM72_fock` to generate simulated values for $w_{s,t}$ and $r_{s,t}$, respectively.

With $w_{s,t}$, $r_{s,t}$, and $k_{s,t}$, you can use {eq}`EqSMM_BM72_bc` to generate simulated values for $c_{s,t}$.

With $\alpha$, $z_{s,t}$, and $k_{s,t}$, you can use {eq}`EqSMM_BM72_prod` to generate simulated values for $y_{s,t}$.

1. Estimate four parameters $(\alpha, \rho,\mu,\sigma)$ given $\beta=0.99$ of the {cite}`BrockMirman:1972` model described by equations {eq}`EqSMM_BM72_eul` through {eq}`EqSMM_BM72_prod` and {eq}`EqSMM_BM72_pf` by SMM. Choose the four parameters to match the following six moments from the 100 periods of empirical data $\{c_t,k_t, w_t, r_t, y_t\}_{t=1}^{100}$ in [`NewMacroSeries.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/smm/NewMacroSeries.txt): $\text{mean}(c_t)$, $\text{mean}(k_t)$, $\text{mean}(c_t/y_t)$, $\text{var}(y_t)$, $\text{corr}(c_t, c_{t-1})$, $\text{corr}(c_t, k_t)$. In your simulations of the model, set $T=100$ and $S=1,000$. Input the bounds to be $\alpha\in[0.01,0.99]$, $\rho\in[-0.99,0.99]$, $\mu\in[5, 14]$, and $\sigma\in[0.01, 1.1]$.
Also, use the identity matrix as your weighting matrix $\textbf{W}=\textbf{I}$ as shown in section {ref}`SecSMM_W_I`. Report your solution $\hat{\theta} = \left(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma}\right)$, the vector of moment differences at the optimum, and the criterion function value. Also report your standard errors for the estimated parameter vector $\hat{\theta} = \left(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma}\right)$ based on the identity matrix for the optimal weighting matrix.
2. Perform the estimation using the two-step estimator for the optimal weighting matrix $\textbf{W}_{2step}$, as shown in section {ref}`SecSMM_W_2step`. Report your solution $\hat{\theta} = \left(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma}\right)$, the vector of moment differences at the optimum, and the criterion function value. Also report your standard errors for the estimated parameter vector $\hat{\theta} = \left(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma}\right)$ based on the two-step optimal weighting matrix $\textbf{W}_{2step}$.
```{exercise-end}
```


(SecSMMFootnotes)=
## Footnotes

The footnotes from this chapter.

[^TruncNorm]: See Section {ref}`SecAppendixTruncNormal` of the Appendix for a description of the truncated normal distribution.
