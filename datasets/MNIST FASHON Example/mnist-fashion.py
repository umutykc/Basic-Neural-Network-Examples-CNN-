#importing libs
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

#loading images and splitting into training and test images
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

#making image color based on 1
train_images, test_images = train_images/255, test_images/255

#creating classes
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal","Shirt", "Sneaker",  "Bag", "Ankle Boot"]

#creating model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#showing model info
model.summary()

#prepearing model for compiling
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#fitting model
model.fit(train_images, train_labels, epochs=5)

#making prediction
prediction = model.predict(test_images)

#visualization predictions
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[test_labels[i]])
    plt.title(classes[np.argmax(prediction[i])])
    plt.show()
