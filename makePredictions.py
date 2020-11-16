import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

class_names = ['T-shirt', 'trousers', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def get_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (28, 28))


images = np.array([get_image("./images/trousers.png"),
                   get_image("./images/ankleboot.png"), get_image('./images/coat.png')])
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show()
images_reshaped = images.reshape(images.shape[0], 28, 28, 1)
images_reshaped = tf.cast(images_reshaped, tf.float32)

model = tf.keras.models.load_model("model.h5")
preds = model.predict(images_reshaped)


def plot_image(prediction, img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    plt.xlabel(f"{class_names[predicted_label]} {round(np.max(prediction)*100, 0)}%",
               color="blue")


def plot_value_array(prediction):
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color('blue')


plt.figure(figsize=(8, 12))
for i in range(3):
    # image
    plt.subplot(3, 2, 2*i+1)
    plot_image(preds[i], images[i])
    # bar chart
    plt.subplot(3, 2, 2*i+2)
    plot_value_array(preds[i])
plt.show()
