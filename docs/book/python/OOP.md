(Chap_OOP)=
# Object Oriented Programming

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

Python is literally a programming language built on objects. Objects are instances of classes. And classes are definitions of objects with their corresponding methods and attributes. Objects are a powerful way to group functionality and attributes in a class that has a limited and common set of characteristics.

An analogy is how life forms are classified by [taxonomic rank](https://en.wikipedia.org/wiki/Taxonomic_rank) going from most general to most specific: domain, kingdom, phylum, class, order, family, genus, and species {cite}`WikiTaxonomicRank`. If you have a model of all the different types of cats, you would probably care about the taxonomic rank *family* of *felidae* or cats. If you were interested in modeling all the different types of mammals that live on land, you might need many different *orders*, with sub-class objects for each *family*, *genus*, and *species* within each order.

Python objects defined as classes have a limited set of attributes that apply to that class in the same way the cat family *felidae* has different attributes than the dog family *canidae*. In the family of `OG-Core` macroeconomic model country calibrations, we have many custom objects defined by classes, the most important of which might be the `parameters` class.

Using objects wisely and efficiently can make your code more readable, easier to modify and use, more scalable, and more interoperable. The iframe below contains a PDF of the BYU ACME open-access lab entitled, "Object-oriented Programming". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_OOP`. {numref}`ExerOOP-acme` below has you work through the problems in this BYU ACME lab. A Python file ([`object_oriented.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/ObjectOriented/.py/object_oriented.py)) template for the problems in this lab is stored in the [`./code/ObjectOriented/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/ObjectOriented) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1dtDaHYhA_7_6vt_uh60CHIPlHf6CA3qf/preview?usp=sharing">
  </iframe>
</div>


(SecOOPExercises)=
## Exercises

```{exercise-start}
:label: ExerOOP-acme
:class: green
```
Read the BYU ACME "[Object-oriented programming](https://drive.google.com/file/d/1dtDaHYhA_7_6vt_uh60CHIPlHf6CA3qf/view?usp=sharing)" lab and complete Problems 1 through 4 in the lab. {cite}`BYUACME_ExceptIO`
```{exercise-end}
```

```{exercise-start}
:label: ExerOOP-defclass
:class: green
```
Define a class called `Specifications` with an attribute that is the rate of time preference `beta` (usually represented by the Greek letter $\beta$). Create two instances of this class, the first called `p1` for `beta=0.96` and the second called `p2` for `beta=0.99`.
```{exercise-end}
```

```{exercise-start}
:label: ExerOOP-attr
:class: green
```
Update the `Specifications` class from {numref}`ExerOOP-defclass` so that it not only allows one to specify the value of `beta` upon instantiation of the class but also checks that `beta` is between 0 and 1.
```{exercise-end}
```

```{exercise-start}
:label: ExerOOP-method
:class: green
```
Modify the `Specifications` class from {numref}`ExerOOP-attr` so that it has a method that prints the value of `beta`.
```{exercise-end}
```

```{exercise-start}
:label: ExerOOP-adjust
:class: green
```
Building off the `Specifications` class in {numref}`ExerOOP-method`, change the input of `beta` to the class so that it is input at an annual rate `beta_annual`. Allow another attribute of the class called `S` that is the number of periods in an economic agent's life. Include a method in the `Specifications` class that adjusts the value of `beta` to represent the discount rate applied per model period. Let each model period be `S/80` years, such that each model period equals one years when `S=80`.
```{exercise-end}
```

```{exercise-start}
:label: ExerOOP-update
:class: green
```
Add a method to the `Specifications` class in {numref}`ExerOOP-adjust` that allows one to update the values of the class attributes `S` and `beta_annual` by providing a dictionary of the form `{"S": 40, "beta_annual": 0.8}`.  Ensure that when the instance is updated, the new `beta` attribute is consistent with the new `S` and `beta_annual`.
```{exercise-end}
```
