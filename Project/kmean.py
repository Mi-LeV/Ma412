#  The objective here is to study unsupervised clustering techniques that allow to group
# the examples in the data set. You should define a metric to decide if your method is
# performing good. In addition, the number of clusters in the data set needs to be determined.
# Any method suggestion not seen mainly in the course will be rewarded.

# The data are clean, and you don't need to pre-process them. 
#But if not, what method would
# you suggest for data processing ? Justify your answer.

#PCA ?
# accuracy relative needs to be 80% for the result to be good

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
X = np.load('data.npy').T
#print(X[:10])
print("Data shape :", X.shape)

fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5)
N = 6
kmean = KMeans(n_clusters=N,init="k-means++")

label = kmean.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("KMENS")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,'\n')
 
for i in u_labels:
    ax1.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax1.legend()

gmm = GaussianMixture(N)
label = gmm.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("gmm")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,"\n")
for i in u_labels:
    ax2.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax2.legend()

agg = AgglomerativeClustering(N)
label = agg.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("agglo")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,"\n")
for i in u_labels:
    ax3.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax3.legend()

bkm = BisectingKMeans(N)
label = bkm.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("Bisecting Kmens")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,"\n")
for i in u_labels:
    ax4.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax4.legend()

dbs = DBSCAN(eps=10, min_samples=8)
label = dbs.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("DBSCAN")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,"\n")
for i in u_labels:
    ax5.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax5.legend()
plt.show()

