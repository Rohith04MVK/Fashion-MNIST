import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
import cv2
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



train_images = train_images.reshape(train_images.shape[0], 28,28,1)
test_images = test_images.reshape(test_images.shape[0], 28,28,1)



model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
	keras.layers.MaxPool2D(pool_size=(2, 2)),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
model.save('model.h5')

'''model = keras.models.load_model('m.model')

def process_image(path):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return cv2.resize(img, (28,28))

images = np.array([process_image('bag.jpg'), process_image('ankle_boot.jpg'),process_image('pants.jpg')])

images_reshaped = images.reshape(images.shape[0], 28,28,1)
images_reshaped = tf.cast(images_reshaped, tf.float64)

preds = model.predict(images_reshaped)

def plot_image(prediction, img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
               100*np.max(prediction),
               ),
                color="blue")
    
def plot_value_array(prediction):
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#888888")
    plt.ylim([0,1])
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color('blue')
    
plt.figure(figsize=(8,12))
for i in range(3):
    # image
    plt.subplot(3, 2, 2*i+1)
    plot_image(preds[i], images[i])
    # bar chart
    plt.subplot(3, 2, 2*i+2)
    plot_value_array(preds[i])
plt.show()  '''
