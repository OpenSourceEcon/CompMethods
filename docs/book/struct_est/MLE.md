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

(Chap_MLE)=
# Maximum Likelihood Estimation

This chapter describes the maximum likelihood estimation (MLE) method. All data and images from this chapter can be found in the data directory ([./data/mle/](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/mle/)) and images directory ([./images/mle/](https://github.com/OpenSourceEcon/CompMethods/tree/main/images/mle/)) for the GitHub repository for this online book.


(SecMLE_GenModel)=
## General characterization of a model and data generating process

Each of the model estimation approaches that we will discuss in this section on Maximum Likelihood estimation (MLE) and in subsequent sections on {ref}`Chap_GMM` (GMM) and {ref}`Chap_SMM` (SMM) involves choosing values of the parameters of a model to make the model match some number of properties of the data. Define a model or a data generating process (DGP) as,

```{math}
    :label: EqMLE_GenMod
    F(x_t, z_t|\theta) = 0
```

where $x_t$ and $z_t$ are variables, $\theta$ is a vector of parameters, and $F()$ is the function expressing the relationship between the variables and parameters.

In richer examples, a model could also include inequalities representing constraints. But this is sufficient for our discussion. The goal of maximum likelihood estimation (MLE) is to choose the parameter vector of the model $\theta$ to maximize the likelihood of seeing the data produced by the model $(x_t, z_t)$.


(SecMLE_GenModel_SimpDist)=
### Simple distribution example

A simple example of a model is a statistical distribution [e.g., the normal distribution $N(\mu, \sigma)$].

```{math}
    :label: EqMLE_GenMod_NormDistPDF
    Pr(x|\theta) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x - \mu)^2}{2\sigma^2}}
```

The probability of drawing value $x_i$ from the distribution $f(x|\theta)$ is $f(x_i|\theta)$. The probability of drawing the following vector of two observations $(x_1,x_2)$ from the distribution $f(x|\theta)$ is $f(x_1|\theta)\times f(x_2|\theta)$. We define the likelihood function of $N$ draws $(x_1,x_2,...x_N)$ from a model or distribution $f(x|\theta)$ as $\mathcal{L}$.

```{math}
    :label: EqMLE_GenMod_NormDistLike
    \mathcal{L}(x_1,x_2,...x_N|\theta) \equiv \prod_{i=1}^N f(x_i|\theta)
```

Because it can be numerically difficult to maximize a product of percentages (one small value can make dominate the entire product), it is almost always easier to use the log likelihood function $\ln(\mathcal{L})$.

```{math}
    :label: EqMLE_GenMod_NormDistLnLike
    \ln\Bigl(\mathcal{L}(x_1,x_2,...x_N|\theta)\Bigr) \equiv \sum_{i=1}^N \ln\Bigl(f(x_i|\theta)\Bigr)
```

The maximum likelihood estimate $\hat{\theta}_{MLE}$ is the following:

```{math}
    :label: EqMLE_GenMod_NormDistMLE
    \hat{\theta}_{MLE} = \theta:\quad \max_\theta \: \ln\mathcal{L} = \sum_{i=1}^N\ln\Bigl(f(x_i|\theta)\Bigr)
```


(SecMLE_GenModel_Econ)=
### Economic example

An example of an economic model that follows the more general definition of $F(x_t, z_t|\theta) = 0$ is {cite}`BrockMirman:1972`. This model has multiple nonlinear dynamic equations, 7 parameters, 1 exogenous time series of variables, and about 5 endogenous time series of variables. Let's look at a simplified piece of that model--the production function--which is commonly used in total factor productivity estimations.

```{math}
    :label: EqMLE_GenMod_EconProdFunc
    Y_t = e^{z_t}(K_t)^\alpha(L_t)^{1-\alpha} \quad\text{where}\quad z_t = \rho z_{t-1} + (1 - \rho)\mu + \varepsilon_t \quad\text{and}\quad \varepsilon_t\sim N(0,\sigma^2)
```

What are the parameters of this model and what are the endogenous variables? If we had data on output $Y_t$, capital $K_t$, and $L_t$, how would we estimate the parameters $\rho$, $\mu$, and $\sigma$? The simplest way I can write this model is $f(Y_t,K_t,L_t|z_0,\rho,\mu,\sigma)=0$.

A maximum likelihood estimation of the parameters $\rho$, $\mu$, and $\sigma$ would either take as data or simulate the total factor productivity series $e^{z_t}$ for all $t$ given the data $Y_t$, $K_t$, and $L_t$, then estimate parameters $\rho$, $\mu$, and $\sigma$ that maximize the likelikhood of those data.

The likelihood of a given data point is determined by $\varepsilon_t = z_t - \rho z_{t-1} - (1 - \rho)\mu \sim N(0,\sigma^2)$. Or in other words the probability of data point $\varepsilon_t$ is $f(z_t - \rho z_{t-1} - (1 - \rho)\mu,\sigma^2$, where $f$ is the normal distribution with mean $z_t - \rho z_{t-1} - (1 - \rho)\mu$ and standard devation $\sigma$.

The likelihood function of all the data is:

```{math}
    :label: EqMLE_GenMod_EconProdFuncLike
    \mathcal{L}\left(z_1,z_2,...z_T|\rho,\mu,\sigma\right) = \prod_{t=2}^T f(z_{t+1},z_t|\rho,\mu,\sigma)
```

The log likelihood function of all the data is:

```{math}
    :label: EqMLE_GenMod_EconProdFuncLnLike
    \ln\Bigl(\mathcal{L}\bigl(z_1,z_2,...z_T|\rho,\mu,\sigma\bigr)\Bigr) = \sum_{t=2}^T \ln\Bigl(f(z_{t+1},z_t|\rho,\mu,\sigma)\Bigr)
```

The maximum likelihood estimate of $\rho$, $\mu$, and $\sigma$ is given by the following maximization problem.

```{math}
    :label: EqMLE_GenMod_EconProdFuncMLE
    (\hat{\rho}_{MLE},\hat{\mu}_{MLE},\hat{\sigma}_{MLE})=(\rho,\mu,\sigma):\quad \max_{\rho,\mu,\sigma}\ln\mathcal{L} = \sum_{t=2}^T \ln\Bigl(f(z_{t+1},z_t|\rho,\mu,\sigma)\Bigr)
```


(SecMLE_DistData)=
## Comparisons of distributions and data

Import some data from the total points earned by all the students in two sections of an intermediate macroeconomics class for undergraduates at an unnamed University in a certain year (two semesters). Let's create a histogram of the data.

```{code-cell} ipython3
:tags: ["remove-output"]

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import requests

# Download and save the data file Econ381totpts.txt as NumPy array
url = ('https://raw.githubusercontent.com/OpenSourceEcon/CompMethods/' +
       'main/data/mle/Econ381totpts.txt')
data_file = requests.get(url, allow_redirects=True)
open('../../../data/mle/Econ381totpts.txt', 'wb').write(data_file.content)
if data_file.status_code == 200:
    # Load the downloaded data into a NumPy array
    data = np.loadtxt('../../../data/mle/Econ381totpts.txt')
else:
    print('Error downloading the file')

# Create a histogram of the data
num_bins = 30
count, bins, ignored = plt.hist(data, num_bins, density=True,
                                edgecolor='k')
plt.title('Intermediate macro scores: 2011-2012', fontsize=15)
plt.xlabel(r'Total points')
plt.ylabel(r'Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"

plt.show()
```

```{figure} ../../../images/mle/Econ381scores_hist.png
---
height: 500px
name: FigMLE_EconScoreHist
---
Intermediate macroeconomics midterm scores over two semesters
```

Now lets code up a parametric distribution that is flexible enough to fit lots of different distributions of test scores, has the properties we would expect from a distribution of test scores, and is characterized by a minimal number of parameters. In this case, we will use a truncated normal distribution.[^TruncNorm]

```{code-cell} ipython3
:tags: []

import scipy.stats as sts


def trunc_norm_pdf(xvals, mu, sigma, cut_lb=None, cut_ub=None):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the truncated normal pdf with mean mu and
    standard deviation sigma. If the cutoff is given, then the PDF
    values are inflated upward to reflect the zero probability on values
    above the cutoff. If there is no cutoff given, this function does
    the same thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
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
```

Let's plot the histogram of the intermediate macroeconomics test scores overlayed by two different truncated nameal distributions, each of which with different arbitrary properties. We want to examine what types of properties make the distribution look more or less like the underlying data.

```{code-cell} ipython3
:tags: ["remove-output"]

# Plot histogram
num_bins = 30
count, bins, ignored = plt.hist(data, num_bins, density=True,
                                edgecolor='k')
plt.title('Intermediate macro scores: 2011-2012', fontsize=15)
plt.xlabel(r'Total points')
plt.ylabel(r'Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"

# Plot smooth line with distribution 1
dist_pts = np.linspace(0, 450, 500)
mu_1 = 380
sig_1 = 150
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_1, sig_1, 0, 450),
         linewidth=2, color='r', label='1: $\mu$=380,$\sigma$=150')
plt.legend(loc='upper left')

# Plot smooth line with distribution 2
mu_2 = 360
sig_2 = 60
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_2, sig_2, 0, 450),
         linewidth=2, color='g', label='2: $\mu$=360,$\sigma$=60')
plt.legend(loc='upper left')

plt.show()
```

```{figure} ../../../images/mle/Econ381scores_2truncs.png
---
height: 500px
name: FigMLE_EconScores2truncs
---
Intermediate macroeconomics midterm scores over two semesters with two arbitrary truncated normal distributions
```

Which distribution will have the biggest log likelihood function? Why?

Let's compute the log likelihood function for this data for both of these distributions.

```{code-cell} ipython3
:tags: []

# Define log likelihood function for the truncated normal distribution
def log_lik_truncnorm(xvals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given truncated
    normal distribution parameters mu, sigma, cut_lb, cut_ub.
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

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

print('Log-likelihood 1: ', log_lik_truncnorm(data, mu_1, sig_1, 0, 450))
print('Log-likelihood 2: ', log_lik_truncnorm(data, mu_2, sig_2, 0, 450))
```

Why is the log likelihood value negative? Which distribution is a better fit according to the Log-likelihood value?

How do we estimate $\mu$ and $\sigma$ by maximum likelihood? What values of $\mu$ and $\sigma$ will maximize the likelihood function?

```{math}
    :label: EqMLE_DistData_maxprob
    (\hat{\mu},\hat{\sigma})_{MLE} = (\mu, \sigma):\quad \max_{\mu,\sigma}\:\ln\,\mathcal{L}=\sum_{i=1}^N\ln\Bigl(f(x_i|\mu,\sigma)\Bigr)
```


(SecMLE_LinReg)=
## Linear regression with MLE

Although linear regression is most often performed using the ordinary least squares (OLS) estimator (see the {ref}`SecBasicEmpLinReg` section of the {ref}`Chap_BasicEmpirMethods` chapter), which is a particular type of generalized method of moments (GMM) estimator (see {ref}`Chap_GMM` chapter), these parameters can also be estimated using maximum likelihood estimation (MLE). A simple regression specification in which the dependent variable $y_i$ is a linear function of two independent variables $x_{1,i}$ and $x_{2,i}$ is the following:

```{math}
    :label: EqMLE_LinReg_eqn
    y_i = \beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i} + \varepsilon_i \quad\text{where}\quad \varepsilon_i\sim N\left(0,\sigma^2\right)
```

If we solve this regression equation for the error term $\varepsilon_i$, we can start to see how we might estimate the parameters of the model by maximum likelihood.

```{math}
    :label: EqMLE_LinReg_eps
    \varepsilon_i = y_i - \beta_0 - \beta_1 x_{1,i} - \beta_2 x_{2,i} \sim N\left(0,\sigma^2\right)
```

The parameters of the regression model are $(\beta_0, \beta_1, \beta_2, \sigma)$. Given some data $(y_i, x_{1,i}, x_{2,i})$ and given some parameter values $(\beta_0, \beta_1, \beta_2, \sigma)$, we could plot a histogram of the distribution of those error terms. And we could compare that empirical histogram to the assumed histogram of the distribution of the errors $N(0,\sigma^2)$. ML estimation of this regression equation is to choose the paramters $(\beta_0, \beta_1, \beta_2, \sigma)$ to make that empirical distribution of errors $\varepsilon_i$ most closely match the assumed distribution of errors $N(0,\sigma^2)$.

Note that estimating a linear regression model using MLE has the flexible property of being able to accomodate any distribution of the error terms, and not just normally distributed errors.


(SecMLE_GBfam)=
## Generalized beta family of distributions

For {numref}`ExercStructEst_MLE_claims`, you will need to know the functional forms of four continuous univariate probability density functions (PDF's), each of which are part of the generalized beta family of distributions. {numref}`Figure %s <FigMLE_GBtree>` below is the generalized beta family of distributions, taken from Figure 2 of {cite}`McDonaldXu:1995`.

```{figure} ../../../images/mle/GBtree.png
---
height: 500px
name: FigMLE_GBtree
---
Generalized beta family of distributions, taken from Fig. 2 of {cite}`McDonaldXu:1995`
```

(SecMLE_GBfam_LN)=
### Lognormal distribution (LN, 2 parameters)

The lognormal distribution (LN) is the distribution of the exponential of a normally distributed variable with mean $\mu$ and standard deviation $\sigma$. If the variable $x_i$ is lognormally distributed $x_i\sim LN(\mu,\sigma)$, then the log of $x_i$ is normally distributed $\ln(x_i)\sim N(\mu,\sigma)$. The PDF of the lognormal distribution is the following.

```{math}
    :label: EqMLE_GBfam_LN
    \text{(LN):}\quad f(x;\mu,\sigma) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{[\ln(x)-\mu]^2}{2\sigma^2}},\quad x\in(0,\infty), \:\mu\in(-\infty,\infty),\: \sigma>0
```

Note that the lognormal distribution has a support that is strictly positive. This is one reason why it is commonly used to approximate income distributions. A household's total income is rarely negative. The lognormal distribution also has a lot of the nice properties of the normal distribution.

(SecMLE_GBfam_GA)=
### Gamma distribution (GA, 2 parameters)

Another two-parameter distribution with strictly positive support is the gamma (GA) distribution. The pdf of the gamma distribution is the following.

```{math}
    :label: EqMLE_GBfam_GA
    \text{(GA):}\quad f(x;\alpha,\beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)}x^{\alpha-1}e^{-\frac{x}{\beta}},\quad x\in[0,\infty), \:\alpha,\beta>0 \\
    \text{where}\quad \Gamma(z)\equiv\int_0^\infty t^{z-1}e^{-t}dt
```

The gamma function $\Gamma(\cdot)$ within the gamma (GA) distribution is a common mathematical function that has a preprogrammed function in most programming languages.

(SecMLE_GBfam_GG)=
### Generalized Gamma distribution (GG, 3 parameters)

The lognormal (LN) and gamma (GA) distributions are both two-parameter distributions and are both special cases of the three-parameter generalized gamma (GG) distribution. The pdf of the generalized gamma distribution is the following.

```{math}
    :label: EqMLE_GBfam_GG
    \text{(GG):}\quad f(x;\alpha,\beta,m) = \frac{m}{\beta^\alpha \Gamma\left(\frac{\alpha}{m}\right)}x^{\alpha-1}e^{-\left(\frac{x}{\beta}\right)^m},\quad x\in[0,\infty), \:\alpha,\beta,m>0 \\
    \text{where}\quad \Gamma(z)\equiv\int_0^\infty t^{z-1}e^{-t}dt
```

The relationship between the generalized gamma (GG) distribution and the gamma (GA) distribution is straightforward. The GA distribution equals the GG distribution at $m=1$.

```{math}
    :label: EqMLE_GBfam_GAtoGG
    GA(\alpha,\beta) = GG(\alpha,\beta,m=1)
```

The relationship between the generalized gamma (GG) distribution and the lognormal (LN) distribution is less straightforward. The LN distribution equals the GG distribution as $\alpha$ goes to zero, $\beta = (\alpha\sigma)^{\frac{2}{\alpha}}$, and $m = \frac{\alpha\mu+1}{\alpha^2\sigma^2}$. See {cite}`McDonaldEtAl:2013` for derivation.

```{math}
    :label: EqMLE_GBfam_LNtoGG
    LN(\mu,\sigma) = \lim_{\alpha\rightarrow 0}GG\left(\alpha,\beta=(\alpha\sigma)^{\frac{2}{\alpha}},m=\frac{\alpha\mu+1}{\alpha^2\sigma^2}\right)
```


(SecMLE_GBfam_GB2)=
### Generalized beta 2 distribution (GB2, 4 parameters)

The last distribution we describe is the generalized beta 2 (GB2) distribution. Like the GG, GA, and LN distributions, it also has a strictly positive support. The PDF of the generalized beta 2 distribution is the following.

```{math}
    :label: EqMLE_GBfam_GB2
    \text{(GB2):}\quad f(x;a,b,p,q) = \frac{a x^{ap-1}}{b^{ap}B(p,q)\left(1 + \left(\frac{x}{b}\right)^a\right)^{p+q}},\quad x\in[0,\infty), \:a,b,p,q>0 \\
    \quad\text{where}\quad B(v,w)\equiv\int_0^1 t^{v-1}(1-t)^{w-1}dt
```

The beta function $B(\cdot,\cdot)$ within the GB2 distribution is a common function that has a preprogrammed function in most programming languages. The three-parameter generalized gamma (GG) distribution is a nested case of the four-parameter generalized beta 2 (GB2) distribution as $q$ goes to $\infty$ and for $a=m$, $b=q^{1/m}\beta$, and $p=\frac{\alpha}{m}$. See {cite}`McDonald:1984`, p. 662 for a derivation.

```{math}
    :label: EqMLE_GBfam_GGtoGB2
    GG(\alpha,\beta,m) = \lim_{q\rightarrow\infty}GB2\left(a=m,b=q^{1/m}\beta,p=\frac{\alpha}{m},q\right)
```

The statistical family tree figure above shows the all the relationships between the various PDF's in the generalized beta family of distributions.


(SecMLE_Exerc)=
## Exercises

```{exercise-start} Health claim amounts and the GB family of distributions
:label: ExercStructEst_MLE_claims
:class: green
```
For this problem, you will use 10,619 health claims amounts from a fictitious sample of households. These data are in a single column of the text file [`claims.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/mle/claims.txt) in the online book repository data folder `data/mle/`. This file is a comma separated text file with no labels. Health claim amounts are reported in US dollars. For this exercise, you will need to use the generalized beta family of distributions shown in {numref}`Figure %s <FigMLE_GBtree>` of Section {ref}`SecMLE_GBfam`.

1. Calculate and report the mean, median, maximum, minimum, and standard deviation of monthly health expenditures for these data. Plot two histograms of the data in which the $y$-axis gives the percent of observations in the particular bin of health expenditures and the $x$-axis gives the value of monthly health expenditures. Use percentage histograms in which the height of each bar is the percent of observations in that bin. In the first histogram, use 1,000 bins to plot the frequency of all the data. In the second histogram, use 100 bins to plot the frequency of only monthly health expenditures less-than-or-equal-to \$800 ($x_i\leq 800$). Adjust the frequencies of this second histogram to account for the observations that you have not displayed ($x_i>800$). That is, the heights of the histogram bars in the second histogram should not sum to 1 because you are only displaying a fraction of the data. Comparing the two histograms, why might you prefer the second one?
2. Using MLE, fit the gamma $GA(x;\alpha,\beta)$ distribution to the individual observation data. Use $\beta_0=Var(x)/E(x)$ and $\alpha_0=E(x)/\beta_0$ as your initial guess. These initial guesses come from the property of the gamma (GA) distribution that $E(x)=\alpha\beta$ and $Var(x)=\alpha\beta^2$. Report your estimated values for $\hat{\alpha}$ and $\hat{\beta}$, as well as the value of the maximized log likelihood function $\ln\mathcal{L}(\hat{\theta})$. Plot the second histogram from part (1) overlayed with a line representing the implied histogram from your estimated gamma (GA) distribution.
3. Using MLE, fit the generalized gamma $GG(x;\alpha,\beta,m)$ distribution to the individual observation data. Use your estimates for $\alpha$ and $\beta$ from part(2), as well as $m=1$, as your initial guess. Report your estimated values for $\hat{\alpha}$, $\hat{\beta}$, and $\hat{m}$, as well as the value of the maximized log likelihood function $\ln\mathcal{L}$. Plot the second histogram from part (1) overlayed with a line representing the implied histogram from your estimated generalized gamma (GG) distribution.
4. Using MLE, fit the generalized beta 2 $GB2(x;a,b,p,q)$ distribution to the individual observation data. Use your estimates for $\alpha$, $\beta$, and $m$ from part (3), as well as $q=10,000$, as your initial guess. Report your estimated values for $\hat{a}$, $\hat{b}$, $\hat{p}$, and $\hat{q}$, as well as the value of the maximized log likelihood function $\ln\mathcal{L}$. Plot the second histogram from part(1) overlayed with a line representing the implied histogram from your estimated generalized beta 2 (GB2) distribution.
5. Perform a likelihood ratio test for each of the estimated in parts (2) and (3), respectively, against the GB2 specification in part (4). This is feasible because each distribution is a nested version of the GB2. The degrees of freedom in the $\chi^2(p)$ is 4, consistent with the GB2. Report the $\chi^2(4)$ values from the likelihood ratio test for the estimated GA and the estimated GG distributions.
6. Using the estimated GB2 distribution from part (4), how likely am I to have a monthly health care claim of more than \$1,000? How does this amount change if I use the estimated GA distribution from part (2)?
```{exercise-end}
```

```{exercise-start} MLE estimation of simple macroeconomic model
:label: ExercStructEst_MLE_BM72
:class: green
```
You can observe time series data in an economy for the following variables: $(c_t, k_t, w_t, r_t)$. Data on $(c_t, k_t, w_t, r_t)$ can be loaded from the file [`MacroSeries.txt`](https://github.com/OpenSourceEcon/CompMethods/blob/main/data/mle/MacroSeries.txt) in the online book repository data folder `data/mle/`. This file is a comma separated text file with no labels. The variables are ordered as $(c_t, k_t, w_t, r_t)$. These data have 100 periods, which are quarterly (25 years). Suppose you think that the data are generated by a process similar to the {cite}`BrockMirman:1972` paper. A simplified set of characterizing equations of the Brock and Mirman model are the following six equations.
```{math}
    :label: EqMLE_BM72_eul
    (c_t)^{-1} - \beta E\left[r_{t+1}(c_{t+1})^{-1}\right] = 0
```
```{math}
    :label: EqMLE_BM72_bc
    c_t + k_{t+1} - w_t - r_t k_t = 0
```
```{math}
    :label: EqMLE_BM72_focl
    w_t - (1-\alpha)e^{z_t}(k_t)^\alpha = 0
```
```{math}
    :label: EqMLE_BM72_fock
    r_t - \alpha e^{z_t}(k_t)^{\alpha-1} = 0
```
```{math}
    :label: EqMLE_BM72_zt
    z_t = \rho z_{t-1} + (1-\rho)\mu + \varepsilon_t \quad\text{where}\quad \varepsilon_t\sim N(0,\sigma^2)
```
```{math}
    :label: EqMLE_BM72_prod
    y_t = e^{z_t}(k_t)^\alpha
```
The variable $c_t$ is aggregate consumption in period $t$, $k_{t+1}$ is total household savings and investment in period $t$ for which they receive a return in the next period (this model assumes full depreciation of capital). The wage per unit of labor in period $t$ is $w_t$ and the interest rate or rate of return on investment is $r_t$. Total factor productivity is $z_t$, which follows an AR(1) process given in {eq}`EqMLE_BM72_zt`. The rest of the symbols in the equations are parameters that must be estimated $(\alpha,\beta,\rho,\mu,\sigma)$. The constraints on these parameters are the following.
\begin{equation*}
  \alpha,\beta \in (0,1),\quad \mu,\sigma > 0, \quad\rho\in(-1,1)
\end{equation*}
Assume that the first observation in the data file variables is $t=1$. Let $k_1$ be the first observation in the data file for the variable $k_t$. Assume that $z_0 = \mu$ so that $z_1= \mu$. Assume that the discount factor is known to be $\beta=0.99$.
1. Use the data $(w_t, k_t)$ and equations {eq}`EqMLE_BM72_focl` and {eq}`EqMLE_BM72_zt` to estimate the four parameters $(\alpha,\rho,\mu,\sigma)$ by maximum likelihood. Given a guess for the parameters $(\alpha,\rho,\mu,\sigma)$, you can use the two variables from the data $(w_t, k_t)$ and {eq}`EqMLE_BM72_focl` to back out a series for $z_t$. You can then use equation {eq}`EqMLE_BM72_zt` to compute the probability of each $z_t\sim N\Bigl(\rho z_{t-1} + (1-\rho)\mu,\sigma^2\Bigr)$. The maximum likelihood estimate $(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma})$ maximizes the likelihood function of that normal distribution of $z_t$'s. Report your estimates and the inverse hessian variance-covariance matrix of your estimates.
2. Now we will estimate the parameters another way. Use the data $(r_t, k_t)$ and equations {eq}`EqMLE_BM72_fock` and {eq}`EqMLE_BM72_zt` to estimate the four parameters $(\alpha,\rho,\mu,\sigma)$ by maximum likelihood. Given a guess for the parameters $(\alpha,\rho,\mu,\sigma)$, you can use the two variables from the data $(r_t, k_t)$ and {eq}`EqMLE_BM72_fock` to back out a series for $z_t$. You can then use equation {eq}`EqMLE_BM72_zt` to compute the probability of each $z_t\sim N\Bigl(\rho z_{t-1} + (1-\rho)\mu,\sigma^2\Bigr)$. The maximum likelihood estimate $(\hat{\alpha},\hat{\rho},\hat{\mu},\hat{\sigma})$ maximizes the likelihood function of that normal distribution of $z_t$'s. Report your estimates and the inverse hessian variance-covariance matrix of your estimates.
3. According to your estimates from part (1), if investment/savings in the current period is $k_t=7,500,000$ and the productivity shock in the previous period was $z_{t-1} = 10$, what is the probability that the interest rate this period will be greater than $r_t=1$. That is, solve for $Pr(r_t>1|\hat{\theta},k_t,z_{t-1})$. [HINT: Use equation {eq}`EqMLE_BM72_fock` to solve for the $z_t=z^*$ such that $r_t = 1$. Then use {eq}`EqMLE_BM72_zt` to solve for the probability that $z_t > z^*$.]
```{exercise-end}
```


(SecMLEfootnotes)=
## Footnotes

The footnotes from this chapter.

[^TruncNorm]: See Section {ref}`SecAppendixTruncNormal` of the Appendix for a description of the truncated normal distribution.
