(Chap_SMM)=
# Simulated Method of Moments

This chapter describes the simulated method of moments (SMM) estimation method. All data and images from this chapter can be found in the data directory ([]./data/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/smm/)) and images directory ([]./images/smm/](https://github.com/OpenSourceEcon/CompMethods/tree/main/images/smm/)) for the GitHub repository for this book.


## The SMM estimator

Simulated method of moments (SMM) is analogous to the generalized method of moments (GMM) estimator. SMM could really be thought of as a particular type of GMM estimator. The SMM estimator chooses model parameters $\theta$ to make simulated model moments match data moments. Seminal papers developing SMM are {cite}`McFadden:1989`, {cite}`LeeIngram:1991`, and {cite}`DuffieSingleton:1993`. Good textbook treatments of SMM are found in {cite}`AddaCooper:2003`, (pp. 87-100) and {cite}`DavidsonMacKinnon:2004`, (pp. 383-394).

In the {ref}`Chap_MaxLikeli` chapter, we used data $x$ and model parameters $\theta$ to maximize the likelihood of drawing that data $x$ from the model given parameters $\theta$.


(SecSMMFootnotes)=
## Footnotes

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
