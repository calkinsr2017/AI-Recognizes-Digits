import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

digits = load_digits()
data = scale(digits.data) #scale all of the features down so that they are between -1 to 1

y = digits.target

k = len(np.unique(y)) # or just do 10. Gives the amount of different classifications dynamically

samples, features = data.shape

clf  = KMeans(n_clusters = k, init = "random", n_init=10)


