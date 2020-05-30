import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    from tensorflow import keras
    import numpy as np 
    import matplotlib.pyplot as plt 
    import pickle 

#This examples is textbassed instead of pictured based data. The input is a movie review and the output is whether or not it is positve or negative

RETRAIN = False


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) #shrinking data a bit

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    return "".join([reverse_word_index.get(i, "?") for i in text])


#########
# Model #
#########

def loadModel(retrain:bool):
    if retrain == False:
        return keras.models.load_model("modelExample2.h5")
    else:
        #Creating the model. Instead of doing it by passing in a list, i am using the add feature.
        model = keras.Sequential()
        model.add(keras.layers.Embedding(88000, 16)) #turns input data into 16 dimensional vectors
        model.add(keras.layers.GlobalAveragePooling1D()) #takes the average of each vector or something
        model.add(keras.layers.Dense(16, activation="relu")) #16 neuron layer
        model.add(keras.layers.Dense(1, activation="sigmoid")) #output layer, positive or negative

        model.summary()

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        x_val = train_data[:10000]
        x_train = train_data[10000:]

        y_val = train_labels[:10000]
        y_train = train_labels[10000:]

        fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
        model.save("modelExample2.h5")
        return model

model = loadModel(RETRAIN)

#results = model.evaluate(test_data, test_labels)

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

with open("movieReview.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''