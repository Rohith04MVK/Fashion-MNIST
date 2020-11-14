import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

# get the data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets\
    .fashion_mnist.load_data()

class_names = ['T-shirt', 'trousers', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


# reshaping
'''train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# define the model
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
# sample output: [0, 0.1, 0, 0.2, 0.7, 0, 0, 0, 0, 0] --> coat
    
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", 
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy: ", test_acc)

model.save('model.h5')'''

# predicting images

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
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100*np.max(prediction),
                                    ),
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


'''# Ploting train data
for i in range(10):
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(f"{class_names[train_labels[i]]}")
    plt.text(0, 0.5,f'{train_images[i].shape}')
    plt.show()'''