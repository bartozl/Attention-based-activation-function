# Combination of activation functions

The aim of this project is to study a new activation function, based on the combination of already known activation functions. In the following paragraphs, different approaches will be briefly explained. The code can be found in the _mixed_activations.py_ file.



## 1. Linear combinator

The activation function is defined as follows:

![](https://latex.codecogs.com/svg.latex?g_j%28s_j%29%20%3D%20%5Csum_i%20%5Calpha_%7Bi%7D%20*%20f_i%28s_j%29)


- ![](https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%7D) parameters to be learned
- ![](https://latex.codecogs.com/svg.latex?f_i) base activation function (e.g. _relu_, _sigmoid_, etc.)
- ![](https://latex.codecogs.com/svg.latex?s)  input
- ![](https://latex.codecogs.com/svg.latex?i) number of neurons of the layer
- ![](https://latex.codecogs.com/svg.latex?j) number of base activation functions




## 2. Non-linear combinator

The activation function is now computed by a Multi Layer Perceptron that takes as input the output of the basic activations (fit with the input).  In pseudo-formula:


![](https://latex.codecogs.com/svg.latex?g_j%28s_j%29%20%3D%20MLP_j%28f_1%28s_j%29%2C%20...%20%2C%20f_i%28s_j%29%29)


## 3. Attention-based combinator

Here, as in the first case, the activation function is the linear combination of the basic activation functions. However, the ![](https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%7D) parameters (*i.e.* the weights of the combination)  are obtained as the output of a MLP. In pseudo-formula:


![](https://latex.codecogs.com/svg.latex?g_j%28s_j%29%20%3D%20%5Csum_i%20%5Calpha_%7Bi%7D%20*%20f_i%28s_j%29)

with

![](https://latex.codecogs.com/svg.latex?%5Calpha_i%20%5Cin%20softmax%28MLP_j%28f_1%28s_j%29%2C%20...%20%2C%20f_i%28s_j%29%29%29)





### Examples of mixed activation functions.



| Linear                      | MLP   | ATT   |
| --------------------------- | ----- | ----- |
| ![](https://ibb.co/Jsw8Cby) | ![]() | ![]() |



**Train and test from scratch:**

```
python feedforward.py
```

**Plot**

```
python plot.py -args
```

where **args** can be: activations, _accuracy_, _table_, _table_max_
