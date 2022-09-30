# PyTorch Implementation of: Error analysis based on inverse modified differential equations for discovery of dynamics using linear multistep methods and deep learning

## Requirements 
* Python 
* torch
* numpy
* matplotlib

## Reproducing the results of the paper
In general all parameters which need to be specified are given in the paper. The code for our experiments is implemented on [PyTorch framework](https://openreview.net/pdf?id=BJJsrmfCZ) with Tesla P40 GPU.

### Running Experiments for Section 5.1:
To train the models, run:
```
./TrainDO.sh
```
After training, run:
```
python DO_plot.py
```

### Running Experiments for Section 5.2:
To train the models, run:
```
./TrainLS.sh
```
After training, run:
```
python LS_plot.py
```

### Running Experiments for Section 5.3:
First generate data,
```
python GO_data.py
```

To train the models, run:
```
./TrainGO.sh
```
After training, run:
```
python GO_plot.py
```
