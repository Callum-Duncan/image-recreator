
# Image recreator

Recreate images using a collection of smaller images


## Installation

clone the repository

```bash
  git clone https://github.com/Callum-Duncan/image-recreator.git
  cd image-recreator
  pip install -r requirements.txt
```
    # Optimizations
`metrics from tqdm`

| Metric                         | Old - Script 2                  | New - Script 13                |
|--------------------------------|---------------------------------|--------------------------------|
| Precalculating Part Info       | 2251 [00:06] 327.72 it/s        | 2251 [00:00] 2273.73 it/s      |
| Extracting Regions             |                                 | 54000 [00:00] 3176522.62 it/s  |
| Calculating Average RGBs       |                                 | 54000 [00:00] 279792.98 it/s    |
| Finding Closest Matching Parts  |                                 | 54000 [00:00] 194946.05 it/s    |
| Placing Closest Parts          | 54000 [00:12] 497.53 it/s       | 54000 [00:00] 710631.33 it/s    |
