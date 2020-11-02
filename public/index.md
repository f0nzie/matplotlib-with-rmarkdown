--- 
title: "Matplotlib with Rmarkdown"
author: "Alfonso R. Reyes"
date: "2020-11-01"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
description: "This is a minimal example of using the bookdown package to write a book. The output format for this example is bookdown::gitbook."
---

# Preface

The goal of **r-test-matplotlib** is testing that Python `matplotlib` and `PyQt5` work seemlessly from within `RStudio 1.2-preview`.

## Motivation
Making `matplotlib` run from within RStudio using the R package `reticulate` and **Python Anaconda** is a very difficult task. **PyQt** is not recognized by RStudio, and requires from the user to provide a path to the **Qt** plugins adding instability to the R-Python installation.

Sometimes, this fix will not work, specially with `PyQt5`.

I found that portable `WinPython 3.6` runs smoothly with RStudio 1.2-preview without crashing it. And without providing a path for PyQt.

## Examples
I have included some working examples of Rmarkdown notebooks with Python `matplotlib` plots.
All of them work without providing a path for PyQt or without crashing RStudio.

## Requisites
1. Install WinPython 3.6.7 64-bit in any folder in your machine. WinPython is portable; it could be installed in a pen drive.

2. Clone this repo under the `notebooks` folder that can be found under `WPy-3670`, the WinPython installation folder.

3. Be sure of loading the `reticulate` package and indicating the Python interpreter.
In the notebook you will see a R chunk with `reticulate::use_python("..\\..\\python-3.6.7.amd64\\python.exe")`. This is key to make R use WinPython.


## WinPython
I downloaded WinPython 3.6 64-bit from [this page](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.7.0/). The specific version I tested was `WinPython64-3.6.7.0Qt5.exe` which can be driectly obtained from [here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.7.0/WinPython64-3.6.7.0Qt5.exe/download). This installation may take around 2 GB of disk space because it is a full fledged Python environment.

The alternative is installing a minimum Python installing `WinPython64-3.6.7.0Zero.exe`. This file can be downloaded from [here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.7.0/WinPython64-3.6.7.0Zero.exe/download). After installation, this zero Python will take around 200 MB. Afterwards, you can install only the packages that you require. I tested this configuration as well, after installing `matplotlib`, `PyQt5`, `Jupyter Notebook`, `pandas`, `numpy`, and few others, taking my imstallation to 400 MB. It run Python `matplotlib` and `PyQt5` from RStudio and `reticulate, flawlessly.



## Bookdown

This is a _sample_ book written in **Markdown**. You can use anything that Pandoc's Markdown supports, e.g., a math equation $a^2 + b^2 = c^2$.

The **bookdown** package can be installed from CRAN or Github:


```r
install.packages("bookdown")
# or the development version
# devtools::install_github("rstudio/bookdown")
```

Remember each Rmd file contains one and only one chapter, and a chapter is defined by the first-level heading `#`.

To compile this example to PDF, you need XeLaTeX. You are recommended to install TinyTeX (which includes XeLaTeX): <https://yihui.org/tinytex/>.


