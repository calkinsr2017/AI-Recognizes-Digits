import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn import linear_model, preprocessing

# K nearest neighbor is a classification algorithm. 
# Will find the K closest values to the unknown. Use an odd value for K
#


#######################
#     Set up data     #
#######################


data = pd.read_csv("car.data")

print(data.head())

#label encoder will take our non numerical data and encode it. EX. low = 0, med = 1, high = 2
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"])) #take the buying column and convert it to the encoded values
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_ = le.fit_transform(list(data["class"]))

predeict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety)) #convert attributes into one big list
y = list(class_)

x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) 

