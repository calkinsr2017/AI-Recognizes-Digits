import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

#This is a simple example of a NN using the glothing dataset on keras. It is a picture based data. You can see what
#The data looks like by uncommenting the plt plots below. 

#######################
#     Set up data     #
#######################


data = keras.datasets.fashion_mnist #built in dataset for different clothings

(train_images, train_labels), (test_images, test_labels) = data.load_data() #Gets the data in tuble form

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#plt.imshow(train_images[7], cmap=plt.cm.binary) #shows the image
#plt.show()

train_images = train_images/255.0  #gets the greyscale value to be between 0 and 1. Its in a numpy array so this works
test_images = test_images/255.0

######################
# Creating the model #
######################


 #takes a list of layers as parameter to create the NN model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #input layer of 784 values. Flattening the input photo
    keras.layers.Dense(128, activation="relu"), #hiden layer 1 of 128 values with relu activation function
    keras.layers.Dense(10,activation="softmax") #output layer
])
#https://keras.io/api/models/model_training_apis/#compile-method
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics = ["accuracy"]) #this kinda sets the parameters for the model

model.fit(train_images, train_labels, epochs=5) #epochs is how many times the model will see the same images just in a different order


prediction = model.predict(test_images)

for i in range(5):
    specific = class_names[np.argmax(prediction[i])]
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction "+ specific)
    plt.show()
    


#test_lost, test_acc = model.evaluate(test_images, test_labels)
