import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#######################
#     Set up data     #
#######################


#Gunna load in datasets for this example

cancer = datasets.load_breast_cancer() #loading a sklearn in house dataset! instead of downloading.

#print(cancer.feature_names)
#print(cancer.target_names)

X = cancer.data 
y = cancer.target

x_train, x_test , y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2) 

classes = ['malignant', 'benign'] #0 and 1 from data will corrispond to this list. 

##########################
# Support Vector Machine #
##########################

#Richard are you reading this? If so I am impressed

svm = False #we can compare svm and KNN

clf = svm.SVC(kernel = "linear", C=2) if svm else KNeighborsClassifier(n_neighbors = 9) #creates the classifier model

clf.fit(x_train, y_train) #trains the model

y_pred = clf.predict(x_test) #tests the model and gives a prediction list

acc = metrics.accuracy_score(y_test, y_pred) #gives the amount of error
print(acc)

