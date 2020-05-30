import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    from tensorflow import keras
    import numpy as np 
    import matplotlib.pyplot as plt 
    import pickle 

#This examples is textbassed instead of pictured based data. The input is a movie review and the output is whether or not it is positve or negative

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000) #shrinking data a bit

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def deode_review(text):
    return "".join([reverse_word_index.get(i, "?") for i in text])


#########
# Model #
#########


#Creating the model. Instead of doing it by passing in a list, i am using the add feature.
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16)) 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) #output layer, positive or negative
