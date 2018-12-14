
<!-- README.md is generated from README.Rmd. Please edit that file -->

# r-test-matplotlib

The goal of r-test-matplotlib is testing that Python `matplotlib` works
seemlessly from within `RStudio 1.2-preview`.

## Motivation

Making `matplotlib` run from within RStudio using the R package
`reticulate` and Python Anaconda is a very difficult task. PyQt is not
recognized by RStudio, and requires from the user to provide a path to
the Qt plugins adding instability to teh R-Python installation.

Sometimes, this fix will not work, specially with PyQt5.

I found that portable WinPython 3.6 runs smoothly with RStudio
1.2-preview without crashing it. And without providing a path for PyQt.

## Examples

I have included some working examples of Rmarkdown notebooks with
matplotlib plots. All of them work without providing a path for PyQt or
without crashing RStudio.

## Requisites

1.  Install WinPython 3.6.7 64-bit in any folder in your machine.
    WinPython is portable; it could be installed in a pen drive.
2.  Clone this repo under the `notebooks` folder that can be found under
    `WPy-3670`, the WinPython installation folder.
3.  Be sure of loading the `reticulate` package and indicating the
    Python interpreter. In the notebook you will see a R chunk with
    `reticulate::use_python("..\\..\\python-3.6.7.amd64\\python.exe")`.
    This is key to make R use WinPython.
