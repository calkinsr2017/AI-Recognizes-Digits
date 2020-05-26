import pandas as pd
import numpy as np 
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#make this a command line argument eventually
RETRAIN = False
GRAPH = True

#######################
#     Set up data     #
#######################

#https://archive.ics.uci.edu/ml/datasets/Student+Performance dataset!

data = pd.read_csv("student-mat.csv", sep=";") #the file is seperated by ; instead of , this is a pandas dataframe
#print(data.head())

#trim data down to what we want it to be
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #attributes
#print(data.head())

predict = "G3" #label, what your looking for

X = np.array(data.drop([predict], 1)) #input
y = np.array(data[predict]) #what we are looking for

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
#this will take existing data and will randomly split the data into training and testing arrays. test_size takes a float
#and tells the function what % of data should go in testing data. AKA 10 of data should be put in testing.
#use this when dataset doesnt have designated training and testing sets
x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) 

###########################################
# Linear regression                       #
# Use when "best fit line" might be used. #
# Looking for correlation                 #
###########################################

#set RETRAIN to true to create and save a new model
def createAndSaveBestModel(retrain: bool, X, y):
    if retrain == False:
        return pickle.load(open("model_accuracy.pickle", "rb"))
    
    best = 0
    for i in range(50):
        x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) 
        #Create the model, fit it and find out its accuracy
        linear = linear_model.LinearRegression() #We need to decide whether or not we want to code this ourselves

        linear.fit(x_train, y_train) #trains the model

        acc = linear.score(x_test, y_test) #This returns the accuracy of our model
        
        if acc > best:
            best = acc
            #Saves the model as a pickle thing... no idea how it works it just does. Pickle is built in to python i guess
            pickle.dump(linear, open("studentmodel.pickle", "wb"))
          
    pickle.dump(best, open("model_accuracy.pickle", "wb"))
    return best

acc = createAndSaveBestModel(RETRAIN, X, y)
print(acc)

#load in the saved trained model
linear = pickle.load(open("studentmodel.pickle", "rb"))

#what are our coefficents for mx+b?
print("Co: \n", linear.coef_) #m1-m5
print("Intercept: \n",  linear.intercept_) #y-intercept

predictions = linear.predict(x_test)

#for x in range(len(predictions)):
    #print(predictions[x], x_test[x], y_test[x]) #predictions should be similar to y_test

#####################
# Graphing results  #
#####################

if GRAPH:
    attribute = "G2" #options are "G1", "G2", "G3", "studytime", "failures", "absences"
    style.use("ggplot")
    pyplot.scatter(data[attribute], data["G3"])
    pyplot.title("attribute vs Final Grade")
    pyplot.xlabel(attribute)
    pyplot.ylabel("Final Grade")
    pyplot.show()