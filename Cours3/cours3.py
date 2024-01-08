import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data_train.txt',delimiter='\t')
df = df[['y', 'x1', 'x2']]
df.plot(x='x1',y='x2',kind='scatter')

df1 = pd.read_csv('data_test.txt',delimiter='\t')
df1 = df[['y', 'x1', 'x2']]
X_test = np.vstack([df1['x1'],df1['x2']]).T
y_test = df1['y']

neigh = KNeighborsClassifier(n_neighbors=30)
X = np.vstack([df['x1'],df['x2']]).T
y = df['y']
neigh.fit(X,y)

print("Mean square error : ", neigh.score(X_test,y_test))
preds = neigh.predict(X_test)
print (confusion_matrix(y_test, preds))