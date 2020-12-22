import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets\
    .fashion_mnist.load_data()

class_names = ['T-shirt', 'trousers', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


for i in range(10):
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(f"{class_names[train_labels[i]]}")
    plt.text(0.01, 0.5, f'{train_images[i].shape}')
    plt.show()
