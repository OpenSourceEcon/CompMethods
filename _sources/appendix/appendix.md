(Chap_Appendix)=
# Appendix

Put Appendix intro here.

(SecAppendixTruncNormal)=
## Truncated normal distribution

The truncated normal distribution with parameters $\mu$ and $\sigma$ and lower-bound cutoff $c_{lb}$ and upper-bound cutoff $c_{ub}$ is simply the normal distribution of values of the random variable $x$ defined only on the interval $x\in[c_{lb}, c_{ub}]$ rather than on the full real line. And the probability distribution function values are upweighted by the probability (less than one) under the normal distribution on the interval $[c_{lb}, c_{ub}]$.
```{math}
    :label: EqAppendix_TruncNorm
    \text{truncated normal:}\quad &f(x|\mu,\sigma,c_{lb},c_{ub}) = \frac{\phi(x|\mu,\sigma)}{\Phi(c_{ub}|\mu,\sigma) - \Phi(c_{ub}|\mu,\sigma)} \\
    &\text{where}\quad \phi(x|\mu,\sigma) \equiv \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{x - \mu}{2\sigma^2}} \\
    &\text{and}\quad \Phi(x|\mu,\sigma) \equiv \int_{-\infty}^x\phi(x|\mu,\sigma) dx
```

The function $\phi(x|\mu,\sigma)$ is the probability distribution function of the normal distribution with mean $\mu$ and variance $\sigma^2$. And the function $\Phi(x|\mu,\sigma)$ is the cummulative distribution function of the normal distribution with mean $\mu$ and variance $\sigma^2$.


(SecAppendixFootnotes)=
## Footnotes

The footnotes from this appendix.
