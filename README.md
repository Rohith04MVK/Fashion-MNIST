[![made-with-python](https://img.shields.io/badge/Made%20with-Python%203.8-ffe900.svg?longCache=true&style=flat-square&colorB=00a1ff&logo=python&logoColor=88889e)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Tensorflow%202.3.1-FFFF00.svg?longCache=true&style=flat-square&colorB=00a1ff&logo=tensorflow&logoColor=FFFF00)](https://www.tensorflow.org/)
# Fashion-MNIST
### Simple fashion accessories recognizing AI
### Using the Fashion Mnist dataset in Tensorflow
# How to run.
### ```conda create -n fashion_mnist python=3.8.5```
### ```conda activate fashion_mnist ```
### ```pip install -r requirements.txt```
### ```python makePredictions.py```
# How to add your own images.
### - Add them to the images folder 
### - Replace the names in predictions file line 16 with your file names
### eg: ```images = np.array([get_image("something.png"), get_image("any.png")])```

# How to it works?
## Neural networks
### Neural Network is essentially a network of mathematical equations. It takes one or more input variables, and by going through a network of equations, results in one or more output variables. You can also say that a neural network takes in a vector of inputs and returns a vector of outputs
### In a neural network, there’s an input layer, one or more hidden layers, and an output layer. The input layer consists of one or more feature variables (or input variables or independent variables) denoted as x1, x2, …, xn. The hidden layer consists of one or more hidden nodes or hidden units. A node is simply one of the circles in the diagram above. Similarly, the output variable consists of one or more output units.
![image](https://miro.medium.com/max/375/1*sTmVItSxeU8nwNfWIuZcqw.png)

## Convolutional Neural Networks(CNN)
### A convolutional neural network (CNN) is a type of neural network that uses a mathematical operation called convolution.
### According to [Wikipidia](https://en.wikipedia.org/wiki/Convolution) Convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other. Thus, CNNs use convolution instead of general matrix multiplication in at least one of their layers.
![image](https://www.researchgate.net/profile/Anjith_George2/publication/303303279/figure/download/fig2/AS:362970388418561@1463550292107/Architecture-of-the-CNN-used.png)

## Rectified linear unit aka ReLu (Activation function)
### Traditionally, some prevalent non-linear activation functions, like sigmoid functions (or logistic) and hyperbolic tangent, are used in neural networks to get activation values corresponding to each neuron. Recently, the ReLu function has been used instead to calculate the activation values in traditional neural network or deep neural network paradigms. The reasons of replacing sigmoid because the ReLu function is able to accelerate the training speed of deep neural networks compared to traditional activation functions since the derivative of ReLu is 1 for a positive input. Due to a constant, deep neural networks do not need to take additional time for computing error terms during training phase.
![image](https://th.bing.com/th/id/OIP.29VH_NiSdoLJ1jUMLrURCAHaC-?pid=ImgDet&rs=1)

## Softmax (Activation function)
### The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution. That is, softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
