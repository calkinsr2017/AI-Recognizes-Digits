import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Neural:
    def __init__(self):
        self.data = tf.keras.datasets.mnist
        
        (x_train, y_train), (x_test, y_test) = self.data.load_data()

        self.x_train = tf.keras.utils.normalize(x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(x_test, axis=1)
        self.y_train = y_train
        self.model = tf.keras.models.Sequential()
        print("initialized")

    #Makes the data binary for testing. My own data will have pixels that are either black or white
    #This will take a WHILE
    def prepareData(self):
        for train in range(len(self.x_train)):
            for row in range(28):
                for x in range(28):
                    if self.x_train[train][row][x] != 0:
                        self.x_train[train][row][x] = 1
        print("prepared")
    
    def createModel(self):
        self.model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        print("created")

    def trainAndSave(self, retrain = False):
        if(retrain):
            self.model.fit(self.x_train, self.y_train, epochs=10)
            self.model.save('mnist_model.h5')

if __name__ == '__main__':
    nn = Neural()
    nn.prepareData()
    nn.createModel()
    nn.trainAndSave(retrain = True)
    print("saved")