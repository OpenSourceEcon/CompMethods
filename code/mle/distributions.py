'''
------------------------------------------------------------------------
This module contains the functions for probability density functions of
continuous PDF's.

This Python module defines the following function(s):
    GA_pdf()
    GG_pdf()
    GB2_pdf()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import scipy.special as spc


'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    This function gives the PDF of the lognormal distribution for xvals
    given mu and sigma

    (LN): f(x; mu, sigma) = (1 / (x * sigma * sqrt(2 * pi))) *
            exp((-1 / 2) * (((log(x) - mu) / sigma) ** 2))
            x in [0, infty), mu in (-infty, infty), sigma > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, data
    mu    = scalar, mean of the ln(x)
    sigma = scalar > 0, standard deviation of ln(x)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals        = (N,) vector, probability of each observation given
                      the parameter values

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = np.float64(((1 / (np.sqrt(2 * np.pi) * sigma * xvals)) *
                          np.exp((-1.0 / 2.0) *
                          (((np.log(xvals) - mu) / sigma) ** 2))))

    return pdf_vals


def GA_pdf(xvals, alpha, beta):
    '''
    --------------------------------------------------------------------
    Returns the PDF values from the two-parameter gamma (GA)
    distribution. See McDonald and Xu (1995).

    (GA): f(x; alpha, beta) = (1 / ((beta ** alpha) *
        spc.gamma(alpha))) * (x ** (alpha - 1)) * (e ** (-x / beta))
    x in [0, infty), alpha, beta > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, values in the support of gamma distribution
    alpha = scalar > 0, gamma distribution parameter
    beta  = scalar > 0, gamma distribution parameter

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.gamma()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, pdf values from gamma distribution
               corresponding to xvals given parameters alpha and beta

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = \
        np.float64((1 / ((beta ** alpha) * spc.gamma(alpha))) *
                   (xvals ** (alpha - 1)) * np.exp(-xvals / beta))

    return pdf_vals


def GG_pdf(xvals, alpha, beta, mm):
    '''
    --------------------------------------------------------------------
    Returns the PDF values from the three-parameter generalized gamma
    (GG) distribution. See McDonald and Xu (1995).

    (GG): f(x; alpha, beta, m) =
        (m / ((beta ** alpha) * spc.gamma(alpha/m))) *
        (x ** (alpha - 1)) * (e ** -((x / beta) ** m))
    x in [0, infty), alpha, beta, m > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, values in the support of generalized gamma (GG)
            distribution
    alpha = scalar > 0, generalized gamma (GG) distribution parameter
    beta  = scalar > 0, generalized gamma (GG) distribution parameter
    mm    = scalar > 0, generalized gamma (GG) distribution parameter

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.gamma()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, pdf values from generalized gamma
               distribution corresponding to xvals given parameters
               alpha, beta, and mm

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = \
        np.float64((mm / ((beta ** alpha) * spc.gamma(alpha / mm))) *
                   (xvals ** (alpha - 1)) *
                   np.exp(-((xvals / beta) ** mm)))

    return pdf_vals


def GB2_pdf(xvals, aa, bb, pp, qq):
    '''
    --------------------------------------------------------------------
    Returns the PDF values from the four-parameter generalized beta 2
    (GB2) distribution. See McDonald and Xu (1995).

    (GB2): f(x; a, b, p, q) = (a * (x ** ((a*p) - 1))) /
        ((b ** (a * p)) * spc.beta(p, q) *
        ((1 + ((x / b) ** a)) ** (p + q)))
    x in [0, infty), alpha, beta, m > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, values in the support of generalized beta 2
            (GB2) distribution
    aa    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    bb    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    pp    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    qq    = scalar > 0, generalized beta 2 (GB2) distribution parameter

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.beta()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, pdf values from generalized beta 2 (GB2)
               distribution corresponding to xvals given parameters aa,
               bb, pp, and qq

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = \
        np.float64((aa * (xvals ** (aa * pp - 1))) / ((bb ** (aa * pp)) *
                   spc.beta(pp, qq) *
                   ((1 + ((xvals / bb) ** aa)) ** (pp + qq))))

    return pdf_vals
