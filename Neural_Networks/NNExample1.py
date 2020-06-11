import warnings  
from PIL import Image, ImageFilter
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    from tensorflow import keras
    import numpy as np 
    import matplotlib.pyplot as plt 
    import pickle
#This is a simple example of a NN using the glothing dataset on keras. It is a picture based data. You can see what
#The data looks like by uncommenting the plt plots below. 

RETRAIN = False

#######################
#     Set up data     #
#######################


data = keras.datasets.mnist #built in dataset for different clothings

(train_images, train_labels), (test_images, test_labels) = data.load_data() #Gets the data in tuble form

class_names = ['zero', 'one', 'two', 'three', 'four', 
                'five', 'six', 'seven', 'eight', 'nine']



train_images = train_images/255.0  #gets the greyscale value to be between 0 and 1. Its in a numpy array so this works
test_images = test_images/255.0

######################
# Creating the model #
######################

def loadModel(retrain:bool):
    if retrain == False:
        return keras.models.load_model("modelExample1.h5")
    else:
        #takes a list of layers as parameter to create the NN model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)), #input layer of 784 values. Flattening the input photo
            keras.layers.Dense(128, activation="relu"), #hiden layer 1 of 128 values with relu activation function
            keras.layers.Dense(10,activation="softmax") #output layer
        ])
        #https://keras.io/api/models/model_training_apis/#compile-method
        model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics = ["accuracy"]) #this kinda sets the parameters for the model

        model.fit(train_images, train_labels, epochs=5) #epochs is how many times the model will see the same images just in a different order
        model.save("modelExample1.h5")
        return model

model = loadModel(RETRAIN)

img = tf.keras.preprocessing.image.load_img("image1.png", target_size = (28,28))
thresh = 200
fn = lambda x : 255 if x > thresh else 0
img = img.convert('L').point(fn, mode='1')
plt.imshow(img)
plt.show()
arr= np.array(img)
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)

print(prediction[0])
print(class_names[np.argmax(prediction[0])])

# for i in range(10, 15):
#     specific = class_names[np.argmax(prediction[i])]
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap = plt.cm.binary)
#     plt.xlabel("Actual: " + class_names[test_labels[i]])
#     plt.title("Prediction "+ specific)
#     plt.show()
    


#test_lost, test_acc = model.evaluate(test_images, test_labels)
