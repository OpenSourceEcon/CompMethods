(Chap_GitIntro)=
# Git and GitHub

This chapter was coauthored by Jason DeBacker and Richard W. Evans.

Two warnings that a seasoned Git and GitHub user should always give a new entrant to this type of version control and code collaboration are the following.
* The learning curve is steep.
* The workflow initially is not intuitive.

These two obstacles seem to work together to make this form of collaboration harder than the sum of their parts initially. However, once you begin collaborating on open source projects or on large-group academic or research projects, you start to see the value of all the different steps, methods, and safeguards invoved with using Git and GitHub. {numref}`Figure %s <FigGitFlowDiag>` below is a diagram of the main pieces and actions in the primary workflow that we advocate in this book. You will notice that a version of this figure is the main image for the book and is also the `favicon` for the tabs of the web pages of the online book. This figure of a Git and GitHub workflow diagram looks complicated, but these actions will become second nature. And following this workflow will save the collaborators time in the long-run.

```{figure} ../images/Git/GitFlowDiag.png
:height: 500px
:align: center
:name: FigGitFlowDiag

Flow diagram of Git and GitHub workflow
```


## Brief definitions

```{prf:definition} Repository
:label: DefRepository

A {term}`repository` or "repo" is a directory containing files that are tracked by a version control system. A local repository resides on a local machine. A {term}`remote` repository resides in the cloud.
```

```{prf:definition} Git
:label: DefGit

{term}`Git` is an {term}`open source` {term}`distributed version control system` (DVCS) software that resides on your local computer and tracks changes and the history of changes to all the files in a directory or {term}`repository`. See the Git website [https://git-scm.com/](https://git-scm.com/) and the [Git Wikipedia entry](https://en.wikipedia.org/wiki/Git) {cite}`GitWiki2020` for more information.
```

```{prf:definition} GitHub
:label: DefGitHub

{term}`GitHub` or [*GitHub.com*](https://github.com/) is a {term}`cloud` {term}`source code management service` platform designed to enable scalable, efficient, and secure version controlled collaboration by linking {term}`local` {term}`Git` version controlled software development by users. *GitHub*'s main business footprint is hosting a collection of millions of version controlled code repositories. In addition to being a platform for {term}`distributed version control system` (DVCS), *GitHub*'s primary features include code review, project management, {term}`continuous integration` {term}`unit testing`, {term}`GitHub actions`, and associated web page (GitHub pages) and documentation hosting and deployment.
```

To be clear at the outset, Git is the version control software that resides on your local computer. It's main functionalities are to track changes in the files in specified directories. But Git also has some functionality to interact with remote repositories. The ineraction between Git and GitHub creates an ideal environment and platform for scaleable collaboration on code among large teams.

## Wide usage
Every year in November, GitHub publishes are report entitled, "The State of the Octoverse", in which they detail the growth and developments in the GitHub community in the most recent year. The most recent [State of the Octoverse](https://github.blog/2022-11-17-octoverse-2022-10-years-of-tracking-open-source/) was published on November 17, 2022 and covered developments from October 1, 2021 to September 30, 2022. Some interesting statistics from that report are the following.

* more than 94 million developers on GitHub
* 85.7 million new repositories in the last year for a total of about 517 million code repositories
* more than 413 million contributions were made to open source projects on GitHub in 2022
* The two most widely used programming languages on GitHub are 1st JavaScript (the language of web dev) and 2nd Python
* more than 90% of Fortune 100 companies use GitHub
* Open source software is now the foundation of more than 90% of the world’s software

Alternatives to GitHub include [GitLab](https://about.gitlab.com/), [Bitbucket](https://bitbucket.org/). Other alternatives are documented in [this June 2020 post](https://www.softwaretestinghelp.com/github-alternatives/) by Software Testing Help. But GitHub has the largest user base and largest number of repositories.


(SecGitBasics)=
## Git and GitHub basics

Create, clone, fork, remote, branch, push, pull, pull request.

Include a discussion of `git pull` vs. `git pull --ff-only` vs. `git pull --rebase`. A good blog post is "[Why You Should Use git pull –ff-only](https://blog.sffc.xyz/post/185195398930/why-you-should-use-git-pull-ff-only)" by Shane at ssfc's Tech Blog.


### Fork a repository and clone it to your local machine

For this example, let the primary repository is [`OG-Core`](https://github.com/PSLmodels/OG-Core) which is in the [PSLmodels](https://github.com/PSLmodels) GitHub organization. This primary repository has a `master` branch that is the lead branch to which we want to contribute and stay up to date.[^MasterMain] If you wanted to contribute to or modify this repository, and you were following the workflow described in {numref}`Figure %s <FigGitFlowDiag>`, you would execute the following three steps.

1. Fork the repository. In your internet browser, go to the main page of the GitHub repository you want to fork (https://github.com/PSLmodels/OG-Core). Click on the "Fork" button in the upper-right corner of the page. This will open a dialogue that confirms the repository owned by you to which you will create the forked copy. This will create an exact copy of the OG-Core repository on your GitHub account or GitHub organization.

2. Clone the repository. In your terminal on your machine, navigate to the directory in which you want your Git repository to reside. Use the `git clone` command plus the URL of the repository on your GitHub account. In the case of my GitHub repository and the OG-Core repository, the command would be the following. Note that you are not cloning the primary repository.

```
DirectoryAboveRepo >> git clone https://github.com/rickecon/OG-Core.git
```

3. Add an `upstream` remote to your fork. Once you have cloned the repository to your local machine, change directories to the new repository on your machine by typing `cd OG-Core` in your terminal. If you type `git remote -v`, you'll see that there is automatically a remote named `origin`. That `origin` name is the name for all the branches on your GitHub account in the cloud associated with the repository. In {numref}`Figure %s <FigGitFlowDiag>`, `origin` represents boxes B and E. You want to add another remote called `upstream` that represents all the branches associated with the primary repository.

```
OG-Core >> git remote add upstream https://github.com/PSLmodels/OG-Core.git
```


### Updating your main or master branch

Let the primary repository is [`OG-Core`](https://github.com/PSLmodels/OG-Core) which is in the [PSLmodels](https://github.com/PSLmodels) GitHub organization. This primary repository has a `master` branch that is the lead branch to which we want to contribute and stay up to date. This repository is represented by box A in {numref}`Figure %s <FigGitFlowDiag>`. You have forked that repository, and your remote fork `master` branch is represented by box B in {numref}`Figure %s <FigGitFlowDiag>` and your local `master` branch is represented by box C.

Suppose that OG-Core has been updated with some pull requests (PRs) that have been merged in. You want to update your remote and local `master` branches (boxes B and C) with the new code from the primary branch (box A).


### Create a development branch to make changes

```
OG-Core >> git checkout -b DevBranchName
```


### Adding, committing, pushing changes to remote repository


### Submit a pull request from your development branch


### Resolve merge conflicts

(SecGitcheatsheet)=
## Git and GitHub Cheat Sheet

About 99% of the commands you'll type in `git` are summarized in the table below:


| Functionality                                               | Git Command                                                      |
|-------------------------------------------------------------|------------------------------------------------------------------|
| See active branch and uncommitted changes for tracked files | `git status -uno`                                                  |
| Change branch                                               | `git checkout <branch name>`                                       |
| Create new branch and change to it                          | `git checkout -b <new branch name>`                                |
| Track file or latest changes to file                        | `git add <filename>`                                               |
| Commit changes to branch                                    | `git commit -m "message describing changes" `                      |
| Push committed changes to remote branch                     | `git push origin <branch name>`                                |
| Merge changes from master into development branch           | `(change working branch to master, then…) git merge <branch name>` |
| Merge changes from development branch into master           | (change to development branch, then…) `git merge master`           |
| List current tags                                           | `git tag`                                                          |
| Create a new tag                                            | `git tag -a v<version number> -m "message with new tag"`           |
| Pull changes from remote repo onto local machine            | `git fetch upstream`                                               |
| Merge changes from remote into active local branch          | `git merge upstream/<branch name>`                                 |
| Clone a remote repository                                   | `git clone <url to remote repo>`                                  |



(SecGitIntroFootnotes)=
## Footnotes

The footnotes from this chapter.

[^MasterMain]: Some primary branches of repositories are called `main` and some are called `master`. Starting in October 2020, GitHub stopped calling the primary branches of repositories `master` and started calling them main. This is due to the potentially offensive or divisive connotations of the term `master`. See {cite}`Wallen:2020`. Repositories with `master` are usually in repos that are older than 2020 and the maintainers have not taken the time to change them.
