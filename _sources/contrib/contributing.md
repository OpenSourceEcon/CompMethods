(Chap_Contrib)=
# Contributor Guide

This chapter details how to contribute to the *Computational Methods for Economists using Python* book and associated repository. The CompMethods project follows the [GitHub workflow](https://guides.github.com/introduction/flow/) and [semantic versioning protocol](http://semver.org/).


## Create an Issue

If you have a suggestion, correction, or addition you want to contribute to the CompMethods project and content, a good first approach is to file an Issue in the repository by going to the [Issues page](https://github.com/OpenSourceEcon/CompMethods/issues) and selecting the green "[New issue](https://github.com/OpenSourceEcon/CompMethods/issues/new)" button. For a productive new issue, please include the following.
* Clear and concise issue title that directly references the key point in your issue
* Clear and concise description of your question, problem, or error
* Error traceback message output or other terminal output
* Include a [minimal reproducible example](https://en.wikipedia.org/wiki/Minimal_reproducible_example)


## Pull requests

This project follows the [GitHub Flow](https://guides.github.com/introduction/flow/). All code contributions are submitted via a [pull request](https://github.com/OpenSourceEcon/CompMethods/pulls) towards the `main` branch. Opening a Pull Request means you are submitting all of the lines of code you have changed in a branch of your fork of this repository that you want to be considered to be merged into the repository. Once a pull request is submitted, project maintainers will review your submission and may ask for changes and clarifications via the pull request message thread. Once the reviewers are satisfied with the submission, they will merge it into the repository and those changes will become part of the project.

### Automatic testing

This project uses GitHub Actions to run automatic tests to make sure that the documentation builds, that the code runs correctly, and that the code is formatted correctly. In your pull requests, you should signify that you have run these tests locally on your machine using the `compmethods-dev` conda environment and successfully running the `make documentation`, `make test`, and `make format` commands. These tests will run automatically in the cloud on every commit to every pull request, but it is helpful for you to successfully run those tests locally on your machine.


### Peer reviews

All pull requests must be reviewed by someone else than their original author, with few exceptions of pull requests from the main model maintainers. To help reviewers, make sure to add to your PR a **clear text explanation** of your changes. In case of changes that break past functionality and connections, you **must** give details about what features were deprecated. You must also provide guidelines to help users adapt their code to be compatible with the new version of the package.

## Project version tracking

This project follows the [semantic versioning protocol](http://semver.org/). Any change impacts the version number, and the version number conveys API compatibility information **only**.

Every pull request submitted to the main branch of the repository should update the `CHANGELOG.md` file as well as update the version number of the project in `setup.py`.

### Patch bump (3rd digit update)

- Typographical and stylistic updates. Small code and data updates.
- Update the third digit of the version number. Ex: Version number would move from 0.0.0 to 0.0.1.

### Minor bump

- Adding a new section, major data, or majore code example to the Jupyter Book
- Update the second digit of the version number. Ex: Version number would move from 0.0.0 to 0.1.0.

### Major bump

- Major update, refactor, or compatibility change.
- Update the first digit of the version number. Ex: Version number would move from 0.0.0 to 1.0.0.
