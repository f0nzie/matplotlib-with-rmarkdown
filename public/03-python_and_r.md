# Sharing data objects


* Always load the Python environment you are sure has the packages


```r
library(reticulate)
use_condaenv("r-torch", required = TRUE)
py_config()
```

```
## python:         /home/msfz751/anaconda3/envs/r-torch/bin/python
## libpython:      /home/msfz751/anaconda3/envs/r-torch/lib/libpython3.7m.so
## pythonhome:     /home/msfz751/anaconda3/envs/r-torch:/home/msfz751/anaconda3/envs/r-torch
## version:        3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]
## numpy:          /home/msfz751/anaconda3/envs/r-torch/lib/python3.7/site-packages/numpy
## numpy_version:  1.19.1
## 
## NOTE: Python version was forced by use_python function
```



```r
reticulate::use_python("..\\..\\python-3.6.7.amd64\\python.exe")
reticulate::py_config()
reticulate::py_available()
```



```r
#R
autos = cars
```



```python
#Python
import numpy
import pandas 

autos_py = r.autos
autos_py['time']=autos_py['dist']/autos_py['speed']
```


```r
#R
plot(py$autos_py)
```

<img src="03-python_and_r_files/figure-html/unnamed-chunk-5-1.png" width="672" />

