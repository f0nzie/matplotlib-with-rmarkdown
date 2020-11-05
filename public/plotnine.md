# Plotnine 0

<div class=decocode><div style="background-color:#4C78DB"><span style="font-size:90%;color:#ffffff"><i class="fab fa-r-project"></i>  <b>R</b></span>

```r
library(reticulate)
reticulate::use_condaenv("r-python")
```

</div><br></div>


<div class=decocode><div style="background-color:#417FB1"><span style="font-size:90%;color:#FFD94C"><i class="fab fa-python"></i>  <b>Python</b></span>

```python
import matplotlib.pyplot as plt
from plotnine import *
from plotnine.data import *
from plotnine.data import mtcars

import numpy as np
import pandas as pd
```

</div><br></div>

<div class=decocode><div style="background-color:#417FB1"><span style="font-size:90%;color:#FFD94C"><i class="fab fa-python"></i>  <b>Python</b></span>

```python
print(mtcars)
```

```
#:>                    name   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
#:> 0             Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
#:> 1         Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
#:> 2            Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
#:> 3        Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3     1
#:> 4     Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3     2
#:> 5               Valiant  18.1    6  225.0  105  2.76  3.460  20.22   1   0     3     1
#:> 6            Duster 360  14.3    8  360.0  245  3.21  3.570  15.84   0   0     3     4
#:> 7             Merc 240D  24.4    4  146.7   62  3.69  3.190  20.00   1   0     4     2
#:> 8              Merc 230  22.8    4  140.8   95  3.92  3.150  22.90   1   0     4     2
#:> 9              Merc 280  19.2    6  167.6  123  3.92  3.440  18.30   1   0     4     4
#:> 10            Merc 280C  17.8    6  167.6  123  3.92  3.440  18.90   1   0     4     4
#:> 11           Merc 450SE  16.4    8  275.8  180  3.07  4.070  17.40   0   0     3     3
#:> 12           Merc 450SL  17.3    8  275.8  180  3.07  3.730  17.60   0   0     3     3
#:> 13          Merc 450SLC  15.2    8  275.8  180  3.07  3.780  18.00   0   0     3     3
#:> 14   Cadillac Fleetwood  10.4    8  472.0  205  2.93  5.250  17.98   0   0     3     4
#:> 15  Lincoln Continental  10.4    8  460.0  215  3.00  5.424  17.82   0   0     3     4
#:> 16    Chrysler Imperial  14.7    8  440.0  230  3.23  5.345  17.42   0   0     3     4
#:> 17             Fiat 128  32.4    4   78.7   66  4.08  2.200  19.47   1   1     4     1
#:> 18          Honda Civic  30.4    4   75.7   52  4.93  1.615  18.52   1   1     4     2
#:> 19       Toyota Corolla  33.9    4   71.1   65  4.22  1.835  19.90   1   1     4     1
#:> 20        Toyota Corona  21.5    4  120.1   97  3.70  2.465  20.01   1   0     3     1
#:> 21     Dodge Challenger  15.5    8  318.0  150  2.76  3.520  16.87   0   0     3     2
#:> 22          AMC Javelin  15.2    8  304.0  150  3.15  3.435  17.30   0   0     3     2
#:> 23           Camaro Z28  13.3    8  350.0  245  3.73  3.840  15.41   0   0     3     4
#:> 24     Pontiac Firebird  19.2    8  400.0  175  3.08  3.845  17.05   0   0     3     2
#:> 25            Fiat X1-9  27.3    4   79.0   66  4.08  1.935  18.90   1   1     4     1
#:> 26        Porsche 914-2  26.0    4  120.3   91  4.43  2.140  16.70   0   1     5     2
#:> 27         Lotus Europa  30.4    4   95.1  113  3.77  1.513  16.90   1   1     5     2
#:> 28       Ford Pantera L  15.8    8  351.0  264  4.22  3.170  14.50   0   1     5     4
#:> 29         Ferrari Dino  19.7    6  145.0  175  3.62  2.770  15.50   0   1     5     6
#:> 30        Maserati Bora  15.0    8  301.0  335  3.54  3.570  14.60   0   1     5     8
#:> 31           Volvo 142E  21.4    4  121.0  109  4.11  2.780  18.60   1   1     4     2
```

</div><br></div>

<div class=decocode><div style="background-color:#417FB1"><span style="font-size:90%;color:#FFD94C"><i class="fab fa-python"></i>  <b>Python</b></span>

```python
# ggplot(data=mpg) + geom_point(mapping=aes(x="displ", y="hwy"))
print(mpg)
```

```
#:>     manufacturer   model  displ  year  cyl       trans drv  cty  hwy fl    class
#:> 0           audi      a4    1.8  1999    4    auto(l5)   f   18   29  p  compact
#:> 1           audi      a4    1.8  1999    4  manual(m5)   f   21   29  p  compact
#:> 2           audi      a4    2.0  2008    4  manual(m6)   f   20   31  p  compact
#:> 3           audi      a4    2.0  2008    4    auto(av)   f   21   30  p  compact
#:> 4           audi      a4    2.8  1999    6    auto(l5)   f   16   26  p  compact
#:> ..           ...     ...    ...   ...  ...         ...  ..  ...  ... ..      ...
#:> 229   volkswagen  passat    2.0  2008    4    auto(s6)   f   19   28  p  midsize
#:> 230   volkswagen  passat    2.0  2008    4  manual(m6)   f   21   29  p  midsize
#:> 231   volkswagen  passat    2.8  1999    6    auto(l5)   f   16   26  p  midsize
#:> 232   volkswagen  passat    2.8  1999    6  manual(m5)   f   18   26  p  midsize
#:> 233   volkswagen  passat    3.6  2008    6    auto(s6)   f   17   26  p  midsize
#:> 
#:> [234 rows x 11 columns]
```

</div><br></div>



The `mpg` dataset:

Plotting couple of variables:


### Facets


```python
# from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
# from plotnine.data import mtcars
# 
# ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)')) + geom_point() + stat_smooth(method='lm') + facet_wrap('~gear'))
```



```python
# import matplotlib.pyplot as plt
# import numpy as np
# from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
# from plotnine.data import mtcars
# 
# (ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)'))
#  + geom_point()
#  + stat_smooth(method='lm')
#  + facet_wrap('~gear'))
```


### Two variables per facet

<div class=decocode><div style="background-color:#417FB1"><span style="font-size:90%;color:#FFD94C"><i class="fab fa-python"></i>  <b>Python</b></span>

```python
# https://plotnine.readthedocs.io/en/stable/generated/plotnine.facets.facet_wrap.html#facet-wrap
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from plotnine import *
# from plotnine.data import *
# 
# mpg.head()
# 
# 
# 
# (
#     ggplot(mpg, aes(x='displ', y='hwy'))
#     + geom_point()
#     + labs(x='displacement', y='horsepower')
# )
# 
# (
#     ggplot(mpg, aes(x='displ', y='hwy'))
#     + geom_point()
#     + facet_wrap('class')
#     + labs(x='displacement', y='horsepower')
# )

# g = ggplot(data=mpg)
# g + geom_point(mapping=aes(x="displ", y="hwy"))
```

</div><br></div>
