from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

np.random.seed(1)
data3 = pd.DataFrame({"data" : np.random.randint(low=1, high=50, size=100) + 50,
                     "target"  : np.random.randint(low=1, high=50, size=100) - 50
                     })
X = data3.data
y = data3.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=1)

plt.scatter(X_train,y_train,color="r")

data = np.vstack([np.array(X_train),np.array(y_train)]).T
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(kmeans.cluster_centers_.T[0],kmeans.cluster_centers_.T[1])
plt.show()
print(kmeans.labels_)
#it has labeled every point with the number of its cluster
print(kmeans.predict([[-3],[-3]]))
