from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


boston = load_boston()
print(boston.data.shape)

X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)

regr = MLPRegressor(random_state=1, max_iter=400).fit(X_train, y_train)
print(regr.score(X_test, y_test))
# score of 0.5 : not that good
parameters = {
    'hidden_layer_sizes' : [(10,),(20,)],
    'activation' : ['tanh', 'relu'],
    'solver' : ['sgd' , 'adam'],
    'alpha' : [0.0001 , 0.05],
    'learning_rate' : ['constant', 'adaptive']
}
clf = GridSearchCV(regr, parameters,n_jobs=-1, cv=3)
print(clf.get_params())
clf.fit(X_test,y_test)
print(clf.best_params_)