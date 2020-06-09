# Combination of activation functions

The aim of this project is to study a new activation function, based on the combination of already known activation functions. In the following paragraphs, different approaches will be briefly explained. The code can be found in the mixed_activations.py file.



## 1. Linear combinator

The activations function is defined as follow:
$
g_j(s_j) = \sum_i α_i * f_i(s_j)
$
Where
$$
α_i = parameters\ to\ be\ learned \\
f_i = base\ activation\  function\ (e.g. "relu",\ "sigmoid",\ etc.)\\
s = input\\
i = number \ of \ neurons \ of \ the \ layer \\
j = number \ of \ base \ activation \ functions
$$




## 2. Non-linear combinator

The activations function is now computed by a Multi Layer Perceptron that take as input the output of the basic activations (fit with the input).  In pseudo-formula:
$$
g_j(s_j) = MLP_J(f_1(s_j), ... , f_i(s_j))
$$




## 3. Attention-based combinator

Here, as in the first case, the activation function is the linear combination of the basic activation functions. However, the α parameters (*i.e.* the weights of the combination)  are obtained as the output of a MLP. In pseudo-formula:
$$
g_j(s_j) = \sum_i α_i * f_i(s_j)
$$
​																					with
$$
α_i \in softmax(MLP_J(f_1(s_j), ... , f_i(s_j)))
$$






**Train and test from scratch:**

```
python feedforward.py
```

**Plot**

```
python plot.py -args
```

where **args** can be: "activations", "accuracy", "table", "table_max"