import tensorflow as tf
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import shutil



class_names = ['Tshirt', 'trousers', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']


def get_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (28, 28))

model = tf.keras.models.load_model("model.h5")
mypath = "./images/{}"
path = "./images/"
onlyfiles = [file for file in listdir(path) if isfile(join(path, file))]

for i in range(len(onlyfiles)):
    images = []
    images = np.array([get_image(f"./images/{onlyfiles[i]}")])
    images_reshaped = images.reshape(images.shape[0], 28, 28, 1)
    images_reshaped = tf.cast(images_reshaped, tf.float32)
    preds = model.predict(images_reshaped)
    image_type = class_names[np.argmax(preds)]

    if isdir(mypath.format(onlyfiles[i].split('.') [0])):
        shutil.move(f'./images/{onlyfiles[i]}', f"./images/{image_type}")
    else:
        print("There is no folder for this")
        print(image_type)
        print(f"./images/{onlyfiles[i]} to ./images/{image_type}")
