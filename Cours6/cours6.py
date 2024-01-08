from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
X,y = load_breast_cancer(return_X_y=True)
pca = PCA(3)
X_fitted = pca.fit(X,y)
print(X_fitted)
print(X_fitted.components_)
#we observe a lot of points near the origin, and some outliers
ax = plt.axes(projection='3d')

#plt.scatter(X_fitted.components_[0],X_fitted.components_[1])
ax.scatter3D(X_fitted.components_[0], X_fitted.components_[1], X_fitted.components_[2],color='red')

plt.show()
# we observe a cluster near the origin, with 3 outliers