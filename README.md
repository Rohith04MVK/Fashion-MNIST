# Fashion-MNIST

[![made-with-python](https://camo.githubusercontent.com/a71f1a20d58a3506dd5f32dcb31461bd5102a0bd33dbf49db9195c589eaca8d7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2532302d2532333134333534432e7376673f267374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465)](https://www.python.org/)[![made-with-tensorflow](https://camo.githubusercontent.com/4058e4719e56be216f2464f47def2f62540a0775acfde94a782f4e1aa9607db7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f54656e736f72466c6f772532302d2532334646364630302e7376673f267374796c653d666f722d7468652d6261646765266c6f676f3d54656e736f72466c6f77266c6f676f436f6c6f723d7768697465)](https://www.tensorflow.org/)[![made-with-keras](https://camo.githubusercontent.com/6282c49f57e9f0ba6b0229225455adc37632dd160625673f1faa03604a0ac42d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4b657261732532302d2532334430303030302e7376673f267374796c653d666f722d7468652d6261646765266c6f676f3d4b65726173266c6f676f436f6c6f723d7768697465)](https://keras.io/)

### Simple fashion accessories recognizing AI using the Fashion Mnist dataset in Tensorflow

# Steps to run it on your machine.
**Note**: You need anaconda installed to use this well.

- Clone the repo : `git clone https://github.com/Rohith04MVK/Fashion-MNIST`
- Create a virtualenv : `conda create -n fashion_mnist python=3.8.5`
- Activate the environment: `conda activate fashion_mnist`
- Install the requirements: `pip install -r requirements.txt`
- Run the main file: `python makePredictions.py`

# How to add your own images.
- Add them to the images folder 
- Replace the names in predictions file line 16 with your file names
###
Example: `images = np.array([get_image("something.png"), get_image("any.png")])`

# The logic and concepts of working:

### Neural networks

Neural Network is essentially a network of mathematical equations. It takes one or more input variables, and by going through a network of equations, results in one or more output variables. You can also say that a neural network takes in a vector of inputs and returns a vector of outputs

In a neural network, there’s an input layer, one or more hidden layers, and an output layer. The input layer consists of one or more feature variables (or input variables or independent variables) denoted as x1, x2, …, xn. The hidden layer consists of one or more hidden nodes or hidden units. A node is simply one of the circles in the diagram above. Similarly, the output variable consists of one or more output units.

![image](https://miro.medium.com/max/375/1*sTmVItSxeU8nwNfWIuZcqw.png)

### Convolutional Neural Networks(CNN)

A convolutional neural network (CNN) is a type of neural network that uses a mathematical operation called convolution.
According to [Wikipedia](https://en.wikipedia.org/wiki/Convolution) Convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other. Thus, CNNs use convolution instead of general matrix multiplication in at least one of their layers.

![image](https://www.researchgate.net/profile/Anjith_George2/publication/303303279/figure/download/fig2/AS:362970388418561@1463550292107/Architecture-of-the-CNN-used.png)

## Rectified linear unit aka ReLu (Activation function)

Traditionally, some prevalent non-linear activation functions, like sigmoid functions (or logistic) and hyperbolic tangent, are used in neural networks to get activation values corresponding to each neuron. Recently, the ReLu function has been used instead to calculate the activation values in traditional neural network or deep neural network paradigms. The reasons of replacing sigmoid because the ReLu function is able to accelerate the training speed of deep neural networks compared to traditional activation functions since the derivative of ReLu is 1 for a positive input. Due to a constant, deep neural networks do not need to take additional time for computing error terms during training phase.

![image](https://ailephant.com/wp-content/uploads/2018/08/ReLU-function-graph-300x234.png)

## Softmax (Activation function)

The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution. That is, softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.

![image](https://th.bing.com/th/id/R1cc898b08e1abb1fc9d3494b19a28595?rik=lxbci3%2bOLVTF4g&riu=http%3a%2f%2f1.bp.blogspot.com%2f_Tndn7IbKcao%2fSyu0vkRlGtI%2fAAAAAAAAAIk%2fTQ-K2fOr9w0%2fs400%2fSigmoidPlot1.png&ehk=%2b3e3aUWb19M3iolTWTGaLwOeAQCrIOa97BLTuavF%2bwg%3d&risl=&pid=ImgRaw)
