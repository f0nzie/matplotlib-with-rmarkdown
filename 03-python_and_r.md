# Sharing data objects


* Always load the Python environment you are sure has the packages


```r
library(reticulate)
use_condaenv("r-python", required = TRUE)
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

<img src="03-python_and_r_files/figure-html/unnamed-chunk-5-1.png" width="90%" style="display: block; margin: auto;" />

