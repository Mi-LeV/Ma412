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
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
X = np.load('data.npy').T
k_range = range(2, 20)
 

distortions = []
inertias = []
 
for k in k_range:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

# Plot the inertia values for each k
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(k_range, inertias, 'bo-')
plt.title('Elbow Method')
ax1.set_ylabel('Inertia')

ax2.plot(k_range, distortions, 'ro-')
plt.xlabel('Number of clusters (k)')
ax2.set_ylabel('Distortions')
plt.show()

# for inertia, the nb of clusters at the elbow seems to be 5<k<10
# for distortion, the nb of clusters at the elbow seems to be 6<k<11
# We shall use K = 6
