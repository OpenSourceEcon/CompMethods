(Chap_SMM)=
# Simulated Method of Moments Estimation

This chapter describes the simulated method of moments (SMM) estimation method. All data and images from this chapter can be found in the data directory ([]./data/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/smm/)) and images directory ([]./images/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/images/smm/)) for the GitHub repository for this book.


(SecSMMestimator)=
## The SMM estimator

Simulated method of moments (SMM) is analogous to the generalized method of moments (GMM) estimator. SMM could really be thought of as a particular type of GMM estimator. The SMM estimator chooses model parameters $\theta$ to make simulated model moments match data moments. Seminal papers developing SMM are {cite}`McFadden:1989`, {cite}`LeeIngram:1991`, and {cite}`DuffieSingleton:1993`. Good textbook treatments of SMM are found in {cite}`AddaCooper:2003`, (pp. 87-100) and {cite}`DavidsonMacKinnon:2004`, (pp. 383-394).

In the {ref}`Chap_MaxLikeli` chapter, we used data $x$ and model parameters $\theta$ to maximize the likelihood of drawing that data $x$ from the model given parameters $\theta$.

```{math}
    :label: EqSMM_MLestimator
    \hat{\theta}_{ML} = \theta:\quad \max_{\theta}\ln\mathcal{L} = \sum_{i=1}^N\ln\Bigl(f(x_i|\theta)\Bigr)
```

In the {ref}`Chap_GMM` chapter, we used data $x$ and model parameters $\theta$ to minimize the distance between model moments $m(x|\theta)$ and data moments $m(x)$.

```{math}
    :label: EqSMM_GMMestimator
    \hat{\theta}_{GMM} = \theta:\quad \min_{\theta}||m(x|\theta) - m(x)||
```

The following difficulties can arise with GMM making it not possible or very difficult.
* The model moment function $m(x|\theta)$ is not known analytically.
* The data moments you are trying to match come from another model (indirect inference, see {cite}`Smith:2020`).
* The model moments $m(x|\theta)$ are derived from *latent variables* that are not observed by the modeler. You only have moments, not the underlying data. See {cite}`LaroqueSalanie:1993`.
* The model moments $m(x|\theta)$ are derived from *censored variables* that are only partially observed by the modeler.
* The model moments $m(x|\theta)$ are just difficult to derive analytically. Examples include moments that include multiple integrals over nonlinear functions as in {cite}`McFadden:1989`.

SMM estimation is simply to simulate the model data $S$ times, and use the average values of the moments from the simulated data as the estimator for the model moments. Let $\tilde{x}=\{\tilde{x}_1,\tilde{x}_2,...\tilde{x}_s,...\tilde{x}_S\}$ be the $S$ simulations of the model data.

```{math}
    :label: EqSMM_AvgSimMoms
    \hat{m}\left(\tilde{x}|\theta\right) = \frac{1}{S}\sum_{s=1}^S m\left(\tilde{x}_s|\theta\right)
```

Once we have an estimate of the model moments $\hat{m}\left(\tilde{x}|\theta\right)$ from our $S$ simulations, SMM estimation is very similar to our presentation of GMM in {ref}`Chap_GMM`. The SMM approach of estimating the parameter vector $\hat{\theta}_{SMM}$ is to choose $\theta$ to minimize some distance measure of the data moments $m(x)$ from the simulated model moments $\hat{m}(\tilde{x}|\theta)$.

```{math}
    :label: EqSMM_SMMestimator
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\: ||\hat{m}(\tilde{x}|\theta)-m(x)||
```

The distance measure $||\hat{m}(\tilde{x}|\theta)-m(x)||$ can be any kind of norm. But it is important to recognize that your estimates $\hat{\theta}_{SMM}$ will be dependent on what distance measure (norm) you choose. The most widely studied and used distance metric in GMM and SMM estimation is the $L^2$ norm or the sum of squared errors in moments. Define the moment error function $e(\tilde{x},x|\theta)$ as the percent difference in the vector of simulated model moments from the data moments.

```{math}
    :label: EqSMM_MomsErrors
    e(\tilde{x},x|\theta) \equiv \frac{\hat{m}(\tilde{x}|\theta)-m(x)}{m(x)} \quad\text{or}\quad \hat{m}(\tilde{x}|\theta)-m(x)
```

It is important that the error function $e(\tilde{x},x|\theta)$ be a percent deviation of the moments (given that none of the data moments are 0). This puts all the moments in the same units, which helps make sure that no moments receive unintended weighting simply due to its units. This ensures that the problem is scaled properly and will suffer from as little as possible ill conditioning.

In this case, the SMM estimator is the following,

```{math}
    :label: EqSMM_SMMestimatorW
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, W \, e(\tilde{x},x|\theta)
```

where $W$ is a $R\times R$ weighting matrix in the criterion function. For now, think of this weighting matrix as the identity matrix. But we will show in Section 2 a more optimal weighting matrix. We call the quadratic form expression $e(\tilde{x},x|\theta)^T \, W \, e(\tilde{x},x|\theta)$ the *criterion function* because it is a strictly positive scalar that is the object of the minimization in the SMM problem statement. The $R\times R$ weighting matrix $W$ in the criterion function allows the econometrician to control how each moment is weighted in the minimization problem. For example, an $R\times R$ identity matrix for $W$ would give each moment equal weighting, and the criterion function would be a simply sum of squared percent deviations (errors). Other weighting strategies can be dictated by the nature of the problem or model.

One last item to emphasize with SMM, which we will highlight in the examples in this notebook, is that the errors that are drawn for the $S$ simulations of the model must be drawn only once so that the minimization problem for $\hat{\theta}_{SMM}$ does not have the underlying sampling changing for each guess of a value of $\theta$. Put more simply, you want the random draws for all the simulations to be held constant so that the only thing changing in the minimization problem is the value of the vector of parameters $\theta$.


(SecSMM_WeightMatW)=
## The Weighting Matrix (W)

In the SMM criterion function in the problem statement above, some weighting matrices $W$ produce precise estimates while others produce poor estimates with large variances. We want to choose the optimal weighting matrix $W$ with the smallest possible asymptotic variance. This is an efficient or optimal SMM estimator. The optimal weighting matrix is the inverse variance covariance matrix of the moments at the optimal moments,

```{math}
    :label: EqSMM_Wopt
    W^{opt} \equiv \Omega^{-1}(\tilde{x},x|\hat{\theta}_{SMM})
```

where $\Omega(\tilde{x},x|\theta)$ is the variance covariance matrix of the moment condition errors $e(\tilde{x},x|\theta)$. The intuition for using the inverse variance covariance matrix $\Omega^{-1}$ as the optimal weighting matrix is the following. You want to downweight moments that have a high variance, and you want to weight more heavily the moments that are generated more precisely.

Notice that this definition of the optimal weighting matrix is circular. $W^{opt}$ is a function of the SMM estimates $\hat{\theta}_{SMM}$, but the optimal weighting matrix is used in the estimation of $\hat{\theta}_{SMM}$. This means that one has to use some kind of iterative fixed point method to find the true optimal weighting matrix $W^{opt}$. Below are some examples of weighting matrices to use.


(SecSMM_W_I)=
### The identity matrix (W=I)

Many times, you can get away with just using the identity matrix as your weighting matrix $W = I$. This changes the criterion function to a simple sum of squared error functions such that each moment has the same weight.

```{math}
    :label: EqSMM_SMMestimatorWI
    \hat{\theta}_{SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, e(\tilde{x},x|\theta)
```

If the problem is well conditioned and well identified, then your SMM estimates $\hat{\theta}_{SMM}$ will not be greatly affected by this simplest of weighting matrices.


(SecSMM_W_2step)=
### Two-step variance-covariance estimator of W


(SecSMM_W_iter)=
### Iterated variance-covariance estimator of W


(SecSMM_W_NW)=
### Newey-West consistent estimator of $\Omega$ and W


(SecSMM_VarCov)=
## The SMM Variance-Covariance Estimator of the Estimated Parameters


(SecSMM_Exmp)=
## Examples


(SecSMMFootnotes)=
## Footnotes

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
