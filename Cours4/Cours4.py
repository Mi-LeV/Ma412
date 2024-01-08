import pandas as pd
from sklearn import svm,metrics,model_selection
import numpy as np

df = pd.read_csv("bill_authentication.csv")
X = df[["Variance","Skewness","Curtosis","Entropy"]]
y = df[["Class"]]
x_tr,x_t,y_tr,y_t = model_selection.train_test_split(X,y,test_size=0.2, random_state=42)


clf = svm.SVC(kernel='linear')
clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_t)
print(metrics.confusion_matrix(y_t,y_pred))

#ex 2

from sklearn import datasets

df = datasets.load_iris()
X = df.data
y = df.target
x_tr,x_t,y_tr,y_t = model_selection.train_test_split(X,y,test_size=0.2, random_state=42)

clf = svm.SVC(kernel='poly',degree=8)
clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_t)
print("poly = ",end="")
print(metrics.confusion_matrix(y_t,y_pred))

print("gauss = ",end="")
clf = svm.SVC(kernel='rbf')
clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_t)
print(metrics.confusion_matrix(y_t,y_pred))

print("sigmoid = ",end="")
clf = svm.SVC(kernel='sigmoid')
clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_t)
print(metrics.confusion_matrix(y_t,y_pred))
# we can see that the most accurate method is the gaussian