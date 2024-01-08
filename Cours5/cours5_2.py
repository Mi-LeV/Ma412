import pandas as pd
from sklearn import svm,metrics,model_selection
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = datasets.load_iris()
X = df.data
y = df.target
x_tr,x_t,y_tr,y_t = model_selection.train_test_split(X,y,test_size=0.2, random_state=42)

regr = MLPRegressor(random_state=1, max_iter=400).fit(x_tr, y_tr)
print(regr.score(x_t, y_t))
# score of 0.5 : not that good
parameters = {
    'hidden_layer_sizes' : [(3,),(10,)],
    'activation' : ['tanh', 'relu'],
    'solver' : ['sgd' , 'adam'],
    'alpha' : [0.0001 , 0.05],
    'learning_rate' : ['constant', 'adaptive']
}
clf = GridSearchCV(regr, parameters,n_jobs=-1, cv=3)
print(clf.get_params())
clf.fit(x_t,y_t)
print(clf.best_params_)