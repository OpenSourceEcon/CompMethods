
(Chap_StructEstPaper)=
# Writing a Structural Estimation Paper

TODO: Finish this section. The content for this section will be taken from my course slides on [creating a structural estimation paper proposal](https://github.com/rickecon/StructEst_W20/blob/master/Projects/ProposalPresent.pdf), and my slides on how to write the following sections of the paper: [data description](https://github.com/rickecon/StructEst_W20/blob/master/Projects/DataSection_slides.pdf), [model description](https://github.com/rickecon/StructEst_W20/blob/master/Projects/ModelDescr_slides.pdf), [estimation section](https://github.com/rickecon/StructEst_W20/blob/master/Projects/EstimResults_slides.pdf), and [conclusion/intro/abstract](https://github.com/rickecon/StructEst_W20/blob/master/Projects/IntroAbsConcl_slides.pdf).


(SecStructEstPaperSections)=
## Sections of a structural estimation project

TODO: Include discussion from project sections and order of project slide in [creating a structural estimation paper proposal](https://github.com/rickecon/StructEst_W20/blob/master/Projects/ProposalPresent.pdf) slides.


(SecStructEstPaperSect_Data)=
### Data description

See [data description](https://github.com/rickecon/StructEst_W20/blob/master/Projects/DataSection_slides.pdf) slides,


(SecStructEstPaperSect_Model)=
### Model description

See [model description](https://github.com/rickecon/StructEst_W20/blob/master/Projects/ModelDescr_slides.pdf) slides.


(SecStructEstPaperSect_Est)=
### Estimation

See [estimation section](https://github.com/rickecon/StructEst_W20/blob/master/Projects/EstimResults_slides.pdf) slides.


(SecStructEstPaperSect_Concl)=
### Conclusion, intro, abstract

See [conclusion/intro/abstract](https://github.com/rickecon/StructEst_W20/blob/master/Projects/IntroAbsConcl_slides.pdf) slides.


(SecStructEstPaperFind)=
## Where/how do I find a project?

TODO: Include discussion from ending slides in [creating a structural estimation paper proposal](https://github.com/rickecon/StructEst_W20/blob/master/Projects/ProposalPresent.pdf) slides. Make sure to include discussion of replication versus original research.


(SecStructEstPaperExerc)=
## Exercises

```{exercise} Create a structural estimation project proposal
:label: ExercStructEst_PaperProposal
:class: green

Create a 5-minute slide presentation of a structural estimation project proposal. You can work alone. However, I recommend you work in a group of (at most) two. The focus of your proposal presentation must be a research question. A good project will have a strong economic theory component. Structural estimation is taking economic theory directly to data. To estimate your model, your project must use GMM, MLE, SMM, or SMLE that you code yourself. I have included a LaTeX beamer style slides template in the [`./code/StrEstPaper/LaTeXtemplates/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/StrEstPaper/LaTeXtemplates) folder of the GitHub repository for this online book if you want to do your slides in LaTeX. Your proposal should include the following requirements.

1. Restrictions
    *  The focus of your proposal presentation must be a research question. No "methods for the sake of methods" projects.
    * You are not allowed to use linear regressions *unless*:
        * it is involved in an indirect inference estimation
        * it is a small subroutine of a bigger model

2. State the research question.
    * What are you trying to learn by using this model?
    * Question should be focused. A narrow question is usually better than a broad question.

3. Describe the model (the data generating process, DGP) $F(x_t,z_t|\theta)=0$
    * What are the endogenous variables $x_t$?
    * What are the exogenous variables $z_t$?
    * What are the parameters $\theta$?
    * Which parameters are estimated $\hat{\theta}_e$?
    * Which parameters are calibrated $\bar{\theta}_c?
    * How does one solve the model given $\theta$?
        * Equations are sufficient (e.g., econometric models)
        * Analytical solution (e.g., behavioral models)
        * Computational solution (e.g., macroeconomic models)

4. Describe the proposed data source $X$
    * How available are the data?
    * Can you show some initial descriptive statistics or visualizations?

5. Describe your proposed estimation strategy $\hat{\theta}$
    * Why did you choose this estimation strategy over alternatives?
    * How will you identify your parameters?
        * MLE: Likelihood function
        * GMM: What moments will you use?

6. Proposal conclusion
    * Restate your research question
    * Outline your hopes and dreams for the project
    * Identify potential shortcomings/alternatives
```

```{exercise} Structural estimation project paper
:label: ExercStructEst_Paper
:class: green

Write a structural estimation paper based on your project proposal from {numref}`ExercStructEst_PaperProposal` using the examples and suggestions from this chapter. I have posted a LaTeX template for a paper in the [`./code/StrEstPaper/LaTeXtemplates/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/StrEstPaper/LaTeXtemplates) folder of the GitHub repository for this online book if you want to do your paper in LaTeX.

1. There is no minimum page requirement, but your paper should be no more than 20 pages long. You can put any extra information in a technical appendix that is not subject to the maximum page requirement.
2. Your paper should be focused on a research question, have a title, clear indication of authors, date, and abstract.
3. You must perform a structural estimation in your paper using one of the following methods: GMM, MLE, SMM, or SMLE that you code yourself.
4. The body of your paper should have the following sections, and you should follow the examples and recommendations for those sections from the corresponding discussions in this chapter.
    * Introduction
    * Data description
    * Model description
    * Estimation
    * Conclusion
```


(SecStructEstPaperFootnotes)=
## Footnotes

The footnotes from this chapter.
