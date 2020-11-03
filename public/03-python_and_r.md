# Sharing data objects


* Always load the Python environment you are sure has the packages

<div class=decocode><div style="background-color:#4C78DB"><span style="font-size:90%;color:#ffffff"><i class="fab fa-r-project"></i>  <b>R</b></span>

```r
# , deco=list(label="R", bc="#4C78DB", tc="#ffffff", icon=list(style="fab", name="r-project"))
library(reticulate)
use_condaenv("r-python", required = TRUE)
```

</div><br></div>


<div class=decocode><div style="background-color:#4C78DB"><span style="font-size:90%;color:#ffffff"><i class="fab fa-r-project"></i>  <b>R</b></span>

```r
#R
autos = cars
```

</div><br></div>


<div class=decocode><div style="background-color:#417FB1"><span style="font-size:90%;color:#FFD94C"><i class="fab fa-python"></i>  <b>Python</b></span>

```python
#Python
import numpy
import pandas 

autos_py = r.autos
autos_py['time']=autos_py['dist']/autos_py['speed']
```

</div><br></div>

<div class=decocode><div style="background-color:#4C78DB"><span style="font-size:90%;color:#ffffff"><i class="fab fa-r-project"></i>  <b>R</b></span>

```r
#R
plot(py$autos_py)
```

<img src="03-python_and_r_files/figure-html/unnamed-chunk-4-1.png" width="90%" style="display: block; margin: auto;" /></div><br></div>


<div class=decocode><div style="background-color:#4C78DB"><span style="font-size:90%;color:#ffffff"><i class="fab fa-r-project"></i>  <b>R</b></span>

```r
reticulate::use_python("..\\..\\python-3.6.7.amd64\\python.exe")
reticulate::py_config()
reticulate::py_available()
```

</div><br></div>
