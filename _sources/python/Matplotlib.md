(Chap_Matplotlib)=
# Matplotlib

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

[Matplotlib](https://matplotlib.org) is Python's most widely used and most basic visualization package.[^Matplotlib1] Some of the other most popular Python visualization packages [Bokeh](http://bokeh.org/), [Plotly](https://plotly.com/), and [Seaborn](https://seaborn.pydata.org/). Of these, Matplotlib is the most general for static images and is what is used on `OG-Core`. Once you have a general idea of how to create plots in Python, that knowlege will generalize (to varying degrees) to the other plotting packages.

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "[Introduction to Matplotlib](https://drive.google.com/file/d/12dnf8tjXBExoQf6W3J5_b52AN27GoBTV/view?usp=sharing)". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Matplotlib1`. {numref}`ExerMatplot-acme1` below has you work through the problems in this BYU ACME lab. A Python file template ([`matplotlib_intro.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib1/matplotlib_intro.py)) and a data file ([`FARS.npy`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib1/FARS.npy)) used in the lab are stored in the [`./code/Matplotlib1/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib1) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/12dnf8tjXBExoQf6W3J5_b52AN27GoBTV/preview?usp=sharing">
  </iframe>
</div>

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "[Pandas 2: Plotting](https://drive.google.com/file/d/1grhP5AcxR9uzvTHSmM4Q4kABM0XENH8r/view?usp=sharing)". In spite of having "Pandas" in the title, we include this lab here in this Matplotlib chapter because all of the plotting uses the Matplotlib package. You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Matplotlib2`. {numref}`ExerMatplot-acme2` below has you work through the problems in this BYU ACME lab. A Jupyter notebook file template ([`matplotlib2.ipynb`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib2/matplotlib2.ipynb)) used in the lab is stored in the [`./code/Matplotlib2/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib2) directory. The [`budget.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas1/budget.csv) and [`crime_data.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas1/crime_data.csv) data files are stored in the [`./data/Pandas1`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas1) directory, which were used in the "[Pandas 1: Introduction](https://drive.google.com/file/d/1t5fjjQXBSIYekZUZIDRvMQOcfCpy8edh/view?usp=sharing)" lab. And the other data file used in this lab [`college.csv`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas3/college.csv) is stored in the [`./data/Pandas3`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas3) directory, which was used in the "[Pandas 3: Grouping](https://drive.google.com/file/d/13DoapcC2whPxSzQQCRaOKv6jow4AkeuZ/view?usp=sharing)" lab.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1grhP5AcxR9uzvTHSmM4Q4kABM0XENH8r/preview?usp=sharing">
  </iframe>
</div>


(SecMatplotlibAnim3D)=
## (Optional): Animations and 3D

This section with its accompanying BYU ACME lab and {numref}`ExerMatplot-acme3` is optional because these plotting skills are used less often and because other plotting packages do a better job of visualization dynamics. That said, this lab is a good one. And the 3D plotting in Matplotlib is fairly good.

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "[Animations and 3D Plotting in Matplotlib](https://drive.google.com/file/d/19y4Uhe4uckSx83duWyELrUz0K-tiYGFc/view?usp=sharing)". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Matplotlib3`. Optional {numref}`ExerMatplot-acme3` below has you work through the problems in this BYU ACME lab. A Jupyter notebook file template ([`animation.ipynb`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib3/animation.ipynb)) used in the lab is stored in the [`./code/Matplotlib3/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Matplotlib3) directory. And two data files used in the lab are stored in the [`./data/Matplotlib3`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Matplotlib3) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/19y4Uhe4uckSx83duWyELrUz0K-tiYGFc/preview?usp=sharing">
  </iframe>
</div>


(SecMatplotlibExercises)=
## Exercises

```{exercise-start}
:label: ExerMatplot-acme1
:class: green
```
Read the BYU ACME "[Introduction to Matplotlib](https://drive.google.com/file/d/12dnf8tjXBExoQf6W3J5_b52AN27GoBTV/view?usp=sharing)" lab and complete Problems 1 through 6 in the lab. {cite}`BYUACME_Matplotlib1`
```{exercise-end}
```

```{exercise-start}
:label: ExerMatplot-acme2
:class: green
```
Read the BYU ACME "[Pandas 2: Plotting](https://drive.google.com/file/d/1grhP5AcxR9uzvTHSmM4Q4kABM0XENH8r/view?usp=sharing)" lab and complete Problems 1 through 4 in the lab. {cite}`BYUACME_Matplotlib2`
```{exercise-end}
```

```{exercise-start} OPTIONAL: Animations nad 3D
:label: ExerMatplot-acme3
:class: green
```
Read the BYU ACME "[Animations and 3D Plotting in Matplotlib](https://drive.google.com/file/d/19y4Uhe4uckSx83duWyELrUz0K-tiYGFc/view?usp=sharing)" lab and complete Problems 1 through 5 in the lab. {cite}`BYUACME_Matplotlib3`
```{exercise-end}
```


```{exercise-start}
:label: ExerMatplot-bar
:class: green
```
Using the country GDP DataFrame you created in Exercise {numref}`ExerPandas-make_df`, collapse these data to find mean GDP per capita by country.  Create a bar plot that shows the means for each of the four countries.
```{exercise-end}
```

```{exercise-start}
:label: ExerMatplot-grouped_bar
:class: green
```
Using same DataFrame as above, create a grouped bar plot that represents the full DataFrame and shows GDP per capita for each country and year.  Group the bar plot so that there is a grouping for each decade and within each group, all four countries are represented.
```{exercise-end}
```

(SecMatplotlibFootnotes)=
## Footnotes

The footnotes from this chapter.

[^Matplotlib1]: Matplotlib's website is https://matplotlib.org.
