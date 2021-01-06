#importing libs
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

#loading images and splitting into training and test images
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

#making image color based on 1
train_images, test_images = train_images/255, test_images/255

#creating classes
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#creating model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#showing model info
model.summary()

#prepearing model for compiling
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

#fitting model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#making prediction
prediction = model.predict(test_images)

#visualization predictions
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[test_labels[i]])
    plt.title(classes[np.argmax(prediction[i][0])])
    plt.show()
