# Combination of activation functions

The aim of this project is to study a new activation function, based on the combination of already known activation functions. In the following paragraphs, different approaches will be briefly explained. The code can be found in the _mixed_activations.py_ file.



## 1. Linear combinator

The activation function is defined as follows:

![](https://latex.codecogs.com/svg.latex?g_n%28s%29%20%3D%20%5Csum_i%20%5Calpha_%7Bi%7D%20*%20f_i%28s%29)


- ![](https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%7D) parameters to be learned
- ![](https://latex.codecogs.com/svg.latex?f_i) base activation function (e.g. _relu_, _sigmoid_, etc.)
- ![](https://latex.codecogs.com/svg.latex?s)  input
- ![](https://latex.codecogs.com/svg.latex?n) number of neurons of the layer
- ![](https://latex.codecogs.com/svg.latex?i) number of base activation functions




## 2. Non-linear combinator

The activation function is now computed by a Multi Layer Perceptron that takes as input the output of the basic activations (fit with the input).  In pseudo-formula:


![](https://latex.codecogs.com/svg.latex?g_n%28s%29%20%3D%20MLP_n%28f_1%28s%29%2C%20...%20%2C%20f_i%28s%29%29)


## 3. Attention-based combinator

Here, as in the first case, the activation function is the linear combination of the basic activation functions. However, the ![](https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%7D) parameters (*i.e.* the weights of the combination)  are obtained as the output of a MLP. In pseudo-formula:


![](https://latex.codecogs.com/svg.latex?g_n%28s%29%20%3D%20%5Csum_i%20%5Calpha_%7Bi%7D%20*%20f_i%28s%29)

with

![](https://latex.codecogs.com/svg.latex?%5Calpha_i%20%5Cin%20softmax%28MLP_n%28f_1%28s%29%2C%20...%20%2C%20f_i%28s%29%29%29)





### Examples of mixed activation functions with _antirelu_, _identity_, _sigmoid_, _tanh_ as base functions.



| Linear  |  ![](https://i.ibb.co/wzF8Ybw/L-AIST.png)  |
| ------- | :----------------------------------------: |
| **MLP** | ![](https://i.ibb.co/8Ks4QJC/MLP-AIST.png) |
| **ATT** | ![](https://i.ibb.co/TbNFH19/ATT-AIST.png) |





#### Train and test

```
python feedforward.py -config config_name.json
```

**config_name** = {_basic_, _linear_, _non_linear_}

#### Plot

```
python plot.py -args -dataset
```

**args** = {activations_, _accuracy_, _table_, _table_max_}

**dataset** = {MNIST, CIFAR10}



#### run_config.json

| parameter     | type                     | value                                                        | description                                                  |
| ------------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| network_type  | integer                  | [1, 2]                                                       | 1: neurons' number divided by 2 for each new layer<br />2: neuron's number is 300 for each new layer |
| nn_layers     | integer                  | [2, inf)                                                     | number of linear layers of the network                       |
| act_fn        | list of lists of strings | ["antirelu", "identity", "relu", "sigmoid", "tanh"]          | basic activations function to combine                        |
| lambda_l1     | list of floats           | (0.0, 0.00000005)                                            | l1 regularization scaling factor                             |
| normalize     | list of strings          | ["None", "Sigmoid", "Softmax"]                               | alpha normalization (only for Linear combinator)             |
| init          | list of strings          | ["None", "random", "uniform", "normal"]                      | alpha initialization (only for Linear combinator)            |
| dataset       | list of strings          | ["MNIST", "CIFAR10"]                                         | available datasets                                           |
| subset        | float                    | (0, 1)                                                       | portion of dataset used                                      |
| epochs        | integer                  | (0, inf)                                                     | number of epochs for training and test                       |
| random_seed   | integer                  | (0, inf)                                                     | allows reproducibility                                       |
| combinator    | list of strings          | ["None", "Linear", "MLP1", "MLP2", "MLP_ATT",  "MLP_ATT_neg"] | available combinators                                        |
| batch_size    | integer                  | (0, inf)                                                     | batch size for training/test                                 |
| alpha_dropout | list of floats           | (0, 1)                                                       | alpha dropout for MLP_ATT combinator                         |



#### Brief code description

- **feedforward.py** is the starting point, it contains the *main*, the *train* and the *test* functions
- **utils.py** contains all the auxiliary functions, both for computations and plotting. The most relevant functions for computation is:
  - *generate_configs*: based on run_config.json, create a list of configurations to be run
- **mixed_activations.py** contains the MIX module (i.e. the core of the project). Also a jit version is implemented
- **modules.py** contains auxiliary modules, such as:
  - *Network*: the neural network used for the experiments. Also a jit version is implemented
  - *MLP1*, *MLP2*, *MLP_ATT*, .... : all small networks needed in the MIX module
- **plot.py** contains all plotting functions. Jit computed models are not plottable.