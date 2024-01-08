import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

X = np.load('data.npy').T
#print(X[:10])
print("Data shape :", X.shape)

for i in np.linspace(2,600,20):
    try:
        dbs = DBSCAN(eps=i, min_samples=19)
        label = dbs.fit_predict(X)
        u_labels = np.unique(label)
        score = silhouette_score(X, label)
        print("DBSCAN",i)
        print("labels : ", len(u_labels))
        print("silhouette higher better",score)
        score = davies_bouldin_score(X, label)
        print("davies lower better",score,"\n")
    except:
        print(i)

# We test for eps = [ 2, 600 ] because for values inferior and superior to these, every value is labeled as noise. 