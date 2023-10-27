(Chap_UnitTesting)=
# Unit Testing

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

As a code base expands and the scripts and modules become more interdependent and interconnected, the probability increases that additions to the code will introduce bugs. And as the code base becomes bigger, the harder it can be to find bugs. One of the primary ways to protect the functionality of a code base from bugs is unit testing.


(SecUnitTestPytest)=
## PyTest

Testing of your source code is important to ensure that the results of your code are accurate and to cut down on debugging time.  Fortunately, `Python` has a nice suite of tools for unit testing. In this section, we will introduce the `pytest` package and show how to use it to test your code.

The iframe below contains a PDF of the BYU ACME open-access lab entitled, "[Unit Testing](https://drive.google.com/file/d/1109ci_tqZz30C2ymf0Hs3UO66l865U0-/view?usp=sharing)". You can either scroll through the lab on this page using the iframe window, or you can download the PDF for use on your computer. See {cite}`BYUACME_UnitTest`. {numref}`ExerTest-acme` below has you work through the problems in this BYU ACME lab. Two Python scripts ([`specs.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnitTest/specs.py) and [`test_specs.py`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnitTest/test_specs.py)) used in the lab are stored in the [`./code/UnitTest/`](https://github.com/OpenSourceEcon/CompMethods/tree/main/code/UnitTest) directory.

<div>
  <iframe id="inlineFrameExample"
      title="Inline Frame Example"
      width="100%"
      height="700"
      src="https://drive.google.com/file/d/1109ci_tqZz30C2ymf0Hs3UO66l865U0-/preview?usp=sharing">
  </iframe>
</div>


(SecUnitTestCodecov)=
## Code coverage

Ideally, one wants to make sure that all of their source code is tested, thereby ensuring it is producing expected results and reducing the potential that new contributions will introduce bugs.  But for any significant code base, it is difficult to know which lines of code are tested and which are.  To get an understanding of what is covered by unit tests, packages like [`coverage.py`](https://coverage.readthedocs.io/en/7.3.2/#) can be used to automatically generate a report of code coverage.  The report will show which lines of code are covered by unit tests and which are not.  This can be useful for identifying parts of the code that need more testing.


(SecUnitTestGHActions)=
## Continuous integration testing and GitHub Actions

When using GitHub to collaborate with others on a code base, one can leverage the ability to use [GitHub Actions](https://github.com/features/actions) to automate unit testing and code coverage reports (as well as other checks on might want to run).  GitHub actions are specified in yaml files and triggered by some set event (e.g., a push, or a pull request, or a chronological schedule).  One of the most effective ways to ensure new contributions are not introducing bugs is to run unit tests and code coverage reports on every push to the repository.  This can be done by creating a GitHub action that runs the unit tests and code coverage report on every push to the repository.  [Codecov](https://about.codecov.io) provides some useful tools for reporting code coverage from unit tests in GitHub Actions.  You can see the actions `OG-Core` uses [here](https://github.com/PSLmodels/OG-Core/tree/master/.github/workflows).  These include unit tests and coverage reports, as well as checks that documentation builds and then is published upon a merge to the `master` branch.


(SecUnitTestExercises)=
## Exercises

```{exercise-start}
:label: ExerTest-acme
:class: green
```
Read the BYU ACME "[Unit Testing](https://drive.google.com/file/d/1109ci_tqZz30C2ymf0Hs3UO66l865U0-/view?usp=sharing)" lab and complete Problems 1 through 6 in the lab. {cite}`BYUACME_UnitTest`
```{exercise-end}
```

```{exercise-start}
:label: ExerTest-assert_value
:class: green
```
In Chapter {ref}`Chap_SciPy`, {numref}`ExerScipy-root-lin`, you wrote wrote a function, and called `SciPy.optimize` to minimize that function. This function had an analytical solution so you could check that SciPy obtained the correct constrained minimum. Now, write a `test_min` function in a module named `test_exercises.py`.  This function should end with an assert statement that the minimum value of the function is equal to the analytical solution.  Then, run the test using `pytest` and make sure it passes. Note, if your wrote the original function for {numref}`ExerScipy-root-lin` in a notebook, copy it over to a module can save it as `exercises.py`.
```{exercise-end}
```

```{exercise-start}
:label: ExerTest-assert_type
:class: green
```
Write another test in your `test_exercises.py` module that uses an assert statement to test that the type of the output of your `test_min` function is a NumPy `ndarray` object.  Then, run the test using `pytest` and make sure it passes.
```{exercise-end}
```

```{exercise-start}
:label: ExerTest-parameterize
:class: green
```
Write a simple function that returns the sum of two digits:
  ```python
  def my_sum(a, b):
    return a + b
  ```
Save this in a module called `exercises.py`.  Now, use the `@pytest.mark.parametrize` decorator to test a function for multiple inputs of `a` and `b`.
```{exercise-end}
```

```{exercise-start}
:label: ExerTest-markers
:class: green
```
Use the `@pytest.mark` decorator to mark one of your tests in `test_exercises.py`.  Then, your tests using `pytest` but in a way that skips tests with the marker you just gave.
```{exercise-end}
```
