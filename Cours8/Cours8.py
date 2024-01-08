from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.data.shape)

X = iris.data[:,:2]
#plt.scatter(X[:,0],X[:,1])
#plt.show()

gmm = GaussianMixture(4)
labels = gmm.fit_predict(X)
colors = {4:'red',3:'black', 2:'green', 1:'blue', 0:'yellow'}
for i in range(len(X)):
    plt.scatter(X[i,0],X[i,1],c=colors[labels[i]])
plt.show()
print("n iter ; ", gmm.n_iter_)
print("value : ",gmm.lower_bound_)