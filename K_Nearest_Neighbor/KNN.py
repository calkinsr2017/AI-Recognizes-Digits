import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

print(data.head)