#  The objective here is to study unsupervised clustering techniques that allow to group
# the examples in the data set. You should define a metric to decide if your method is
# performing good. In addition, the number of clusters in the data set needs to be determined.
# Any method suggestion not seen mainly in the course will be rewarded.

# The data are clean, and you don't need to pre-process them. 
#But if not, what method would
# you suggest for data processing ? Justify your answer.


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
print("KMEANS")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,'\n')
 
for i in u_labels:
    ax1.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax1.legend()
ax1.title.set_text('K-means')

gmm = GaussianMixture(N)
label = gmm.fit_predict(X)
u_labels = np.unique(label)
score = silhouette_score(X, label)
print("GMM")
print("silhouette higher better",score)
score = davies_bouldin_score(X, label)
print("davies lower better",score,"\n")
for i in u_labels:
    ax2.scatter(X[label == i , 7] , X[label == i , 8] , label = i)
ax2.legend()
ax2.title.set_text('GMM')

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
ax3.title.set_text('Agglomerative')

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
ax4.title.set_text('Bisecting K-Means')

dbs = DBSCAN(eps=500, min_samples=19)
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
ax5.title.set_text('DBSCAN')

plt.show()

# The Silhouette Score measures the quality of clustering by computing the average distance between a data point and all other points within the same cluster (intra-cluster distance) and 
# comparing it to the average distance between that data point and all points in the nearest neighboring cluster (inter-cluster distance).
# The silhouette score ranges from -1 to +, so a score close to +1 indicates that the data point is well-clustered and lies far away from neighbouring clusters.
# The average silhouette score for all samples is used as an overall measure of the clustering quality. A higher average silhouette score implies better-defined clusters.
# The Davies-Bouldin Index measures the average similarity between each cluster and its most similar cluster, taking into account both intra-cluster and inter-cluster distances.
# A lower Davies-Bouldin Index indicates better clustering; smaller values suggest better separation between clusters.
# The aim is to maximize the Silhouette Score and minimize the Davies-Bouldin Index for better-defined and well-separated clusters

# We can see that for a cluster size of 6, K-Means and the Agglomeration clustering methods perform the best on the dataset.
# We will also test the use of DBSCAN, which doesnt have a fixed number of clusters ( which includes noise ) but uses density of the data. To tune it,
# a heuristic method used is to first defines the minimum points based on the number of dimensions + 1 ( here 18 ).
# Then, we try the clustering method on a range of epsilon from 2 to 600 to try to match the number of clusters found by the elbow method.
# We find that at eps = 500, we have the numbers of clusters N = 4 (excluding noise). 

# Using this method and comparing it to the others with its Silhouette and Davies-Bouldin, we find that it is the best scoring method of them all, using only 4 clusters
# ( so there is no overfitting ).
# Thus, we can conclude that it is the best method for our dataset. 