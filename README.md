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


## References
[1] [MultistepNNs](https://github.com/maziarraissi/MultistepNNs).

[2] [learner](https://github.com/jpzxshi/learner).

[3] Qiang Du, Yiqi Gu, Haizhao Yang, Chao Zhou. [The Discovery of Dynamics via Linear Multistep Methods and Deep Learning: Error Estimation](https://arxiv.org/abs/2103.11488). SIAM Journal on Numerical Analysis, 60(4):2014–2045, (2022).

[4] Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. [Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems](https://arxiv.org/abs/1801.01236). arXiv preprint arXiv:1801.01236 (2018).

[5] Aiqing Zhu, Sidi Wu, Yifa Tang. [Error analysis based on inverse modified differential equations for discovery of dynamics using linear multistep methods and deep learning](https://arxiv.org/abs/2209.12123). arXiv preprint arXiv: 2209.12123 (2022).
