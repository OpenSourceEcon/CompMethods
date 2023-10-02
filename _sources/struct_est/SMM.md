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

Define the moment error vector $e(\tilde{x},x|\theta)$ as the $R\times 1$ vector of average moment error functions $e_r(\tilde{x},x|\theta)$ of the $r$th average moment error. We can define the $r$th average moment error as the percent difference in the average simulated $r$th moment value $\hat{m}_r(\tilde{x}|\theta)$ from the $r$th data moment $m_r(x)$.

```{math}
    :label: EqSMM_rMomError
    e_r(\tilde{x},x|\theta) \equiv \frac{\hat{m}_r(\tilde{x}|\theta)-m_r(x)}{m_r(x)} \quad\text{or}\quad \hat{m}_r(\tilde{x}|\theta)-m_r(x)
```

It is important that the error function $e(\tilde{x},x|\theta)$ be a percent deviation of the moments (given that none of the data moments are 0). This puts all the moments in the same units, which helps make sure that no moments receive unintended weighting simply due to its units. This ensures that the problem is scaled properly and will suffer from as little as possible ill conditioning.

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
    :label: EqSMM_estW_2step_1
    \hat{\theta}_{1,SMM}=\theta:\quad \min_{\theta}\:e(\tilde{x},x|\theta)^T \, I \, e(\tilde{x},x|\theta)
```

Because we are simulating data, we can generate an estimator for the variance covariance matrix of the moment error vector $\hat{\Omega}$ using just the simulated data moments and the data moments. This $E(\tilde{x},x|\theta)$ matrix represents the contribution of the $s$th simulated moment to the $r$th moment error. Define $E(\tilde{x},x|\theta)$ as the $R\times S$ matrix of moment error functions from each simulation.

```{math}
    :label: EqSMM_estW_errmat_lev_1
    ?
```

text

```{math}
    :label: EqSMM_estW_errmat_pct_1
    ?
```

text


(SecSMM_W_iter)=
### Iterated variance-covariance estimator of W


(SecSMM_W_NW)=
### Newey-West consistent estimator of $\Omega$ and W


(SecSMM_VarCov)=
## The SMM Variance-Covariance Estimator of the Estimated Parameters


(SecSMM_Exmp)=
## Examples


(SecSMM_Exerc)=
## Exercises

```{exercise-start}
:label: ExercStructEst_SMM_BM72
```
**Estimating the Brock and Mirman (1972) model by SMM.** You can observe time series data in an economy for the following variables: $(c_t, k_t, w_t, r_t, y_t)$. The data can be loaded from the file [`NewMacroSeries.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/smm/NewMacroSeries.txt) in the online book repository data folder `data/smm/`. This file is a comma separated text file with no labels. The variables are ordered as $(c_t, k_t, w_t, r_t, y_t)$. These data have 100 periods, which are quarterly (25 years). Suppose you think that the data are generated by a process similar to the {cite}`BrockMirman:1972` paper. A simplified set of characterizing equations of the Brock and Mirman model are the following six equations.
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

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
