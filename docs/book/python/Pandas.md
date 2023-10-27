(Chap_Pandas)=
# Pandas

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

Pandas is to data wrangling and analysis in Python what {ref}`Chap_NumPy` is to numerical methods in Python. Pandas is Python's primary data analysis package.[^Pandas1] Its name is derived from the econometric term, "panel data". The Pandas package was originially developed in 2008 by Wes McKinney while at global investment firm AQR Capital Management.[^Pandas2] The Python Pandas package became open source in 2009, and Pandas became a NumFOCUS sponsored project in 2015.

The primary Python object in Pandas is the DataFrame ([`pandas.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)). A Pandas DataFrame is similar to the R programming language's dataframe.[^PandasR] The dataframe is a two-dimensional data object that often include rows that serve as observations, columns that serve as variables, advanced date functions, and rich multi-layered indexing capability. Pandas also includes rich functionality for reading in data, saving and exporting data, data cleaning and munging, data description, and data manipulation, selection, and grouping.

Pandas also has a Series object ([`pandas.Series`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)), that represents a single data series, often a single variable from a DataFrame. The operations on and attributes of a Pandas Series object are similar to those of the DataFrame.

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "Pandas 1: Introduction". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Pandas1`. {numref}`ExerPandas-acme1` below has you work through the problems in this BYU ACME lab. The data files used in this lab are stored in the [`./data/Pandas1/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas1) directory. A Jupyter notebook file template ([`pandas1.ipynb`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Pandas1/pandas1.ipynb)) used in the lab is stored in the [`./code/Pandas1/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Pandas1) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1t5fjjQXBSIYekZUZIDRvMQOcfCpy8edh/preview?usp=sharing">
  </iframe>
</div>

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "Pandas 3: Grouping". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_Pandas3`. {numref}`ExerPandas-acme2` below has you work through the problems in this BYU ACME lab. The data files used in this lab are stored in the [`./data/Pandas3/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/data/Pandas3) directory. A Jupyter notebook file template ([`pandas3.ipynb`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Pandas3/pandas3.ipynb)) used in the lab is stored in the [`./code/Pandas3/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/Pandas3) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/13DoapcC2whPxSzQQCRaOKv6jow4AkeuZ/preview?usp=sharing">
  </iframe>
</div>


(SecPandasExercises)=
## Exercises

```{exercise-start}
:label: ExerPandas-acme1
:class: green
```
Read the BYU ACME "[Pandas 1: Introduction](https://drive.google.com/file/d/1t5fjjQXBSIYekZUZIDRvMQOcfCpy8edh/view?usp=sharing)" lab and complete Problems 1 through 6 in the lab. {cite}`BYUACME_Pandas1`
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-acme2
:class: green
```
Read the BYU ACME "[Pandas 3: Grouping](https://drive.google.com/file/d/13DoapcC2whPxSzQQCRaOKv6jow4AkeuZ/view?usp=sharing)" lab and complete Problems 1 through 5 in the lab. {cite}`BYUACME_Pandas3`
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-make_df
:class: green
```
Consider the following GDP per capita data (in constant 2011$, source: [Maddison Project Database](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2020?lang=en)):
|                | IND | MYS | USA | ZAF|
|----------------|----|----|----|----|
| 1990        |  2,087  | 8,179   | 36,982   |  6,111  |
| 2000         |  2,753  | 13,475   |  45,886  |  7,583  |
| 2010         | 4,526   |  18,574  |  49,267  | 11,319   |
| 2018 |   6,806 |  24,842  | 55,335   | 12,166   |

Create a dictionary with keys `Year`, `IND`, `MYS`, `USA`, and `ZAF` and values that are lists of the GDP per capita data for each country.  Create a DataFrame named `df` from this dictionary.  Print the DataFrame.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-inspect
:class: green
```
Inspect this data frame.  Print `df.head(3)`.  Print `df.tail(3)`.  Get a list of column names with the `keys` method.  Finally, use the  `describe` method to print descriptive statistics.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-index
:class: green
```
Pandas DataFrames use an index to keep track of rows. Note the default index in a DataFrame `df` are integers for each row. Change the index so the year is the index value. Print the updated DataFrame `df`.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-reshape
:class: green
```
In this exercise reshape your DataFrame `df` from {numref}`ExerPandas-index` into a long panel format with a [`MultiIndex`](https://pandas.pydata.org/docs/user_guide/advanced.html) for the columns. The first level of the `MultiIndex` should be the country name and the second level should be the year. The values should be the GDP per capita. To do this, use the [`pivot_table`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html) or [`stack`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html) methods of the DataFrame class. Please print the resulting DataFrame.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-groupby
:class: green
```
Create a new variable that is the growth rate in GDP per capita from the prior period measure. To do this, use [`groupby`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) to find growth rate for each country over the sample.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-print_tables
:class: green
```
The DataFrame object has several methods to help output a formatted table suitable for reports or presentations. Use one of these methods to print a DataFrame formatted as a [markdown](https://www.markdownguide.org/basic-syntax/) table.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-read
:class: green
```
In most cases, you are likely to use a DataFrame as a container for a large dataset, not something simple that you can enter by manually as we did above. Pandas has [several methods](https://pandas.pydata.org/docs/user_guide/io.html) to read in data from files are various formats. Let's use one of these methods to read in some population data extracted from the [United Nations' World Population Prospects](https://population.un.org/wpp/). Note that Pandas will download these data for you if you have a URL to the data file. The URL for these data on South Africa's population is: [https://raw.githubusercontent.com/EAPD-DRB/OG-ZAF/main/ogzaf/data/demographic/un_zaf_pop.csv](https://raw.githubusercontent.com/EAPD-DRB/OG-ZAF/main/ogzaf/data/demographic/un_zaf_pop.csv). Please read in these data (Note: the separator is the verical bar ("|") and the header is on the second line (in Python this has index 1, so you'll want to use the argument `header=1`)). Print the first 5 rows of the DataFrame.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-subset
:class: green
```
Now we'll select a subset of this DataFrame. Please create a new DataFrame called `zaf_pop` that contains only the columns `AgeId`, `Value` and only rows where `SexId=3` (i.e., both sexes are included), `TimeLabel=2021` (i.e., only values for the year 2021). Print the first 5 rows of the DataFrame.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-new_var
:class: green
```
With your new `zaf_pop` DataFrame, rename the column `Value` to `Count`. Create a new variable in the DataFrame called `Density` that is the fraction of the total population for each age. Print the first 5 rows of the DataFrame.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-plot
:class: green
```
Use the Pandas DataFrame [`plot`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html) method to plot the population density across age for South Africa.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-merge
:class: green
```
 It is often the case that we need to combine more than one dataset. Pandas offers a few options to do this, including the [`merge`] and [`join`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html) methods of the DataFrame class. Let's test this, but reading in the original data again, and finding the population of women in 2021. Then use `merge` or `join` to combine the `zaf_pop` and `zaf_female_pop` DataFrames. Plot the density of women and the overall population together.
```{exercise-end}
```

```{exercise-start}
:label: ExerPandas-save
:class: green
```
Save your final DataFrame to your hard drive as a comma separated values `.csv` format file.
```{exercise-end}
```


(SecPandasFootnotes)=
## Footnotes

The footnotes from this chapter.

[^Pandas1]: The website for Pandas is https://pandas.pydata.org/.

[^Pandas2]: See the "About" page on the Pandas website (https://pandas.pydata.org/about/) as well as the Pandas Wikipedia article {cite}`PandasWiki`.

[^PandasR]: The Pandas online documentation has a page that gives a correspondence between Pandas Dataframe functionality and R dataframe functionality (https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html).
