from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso
df = pd.read_csv('prostate.data.txt', sep="\t")

x_tr = df.iloc[:50,1:5]
y_tr = df.loc[:49,"lpsa"]

x_t = df.iloc[50:,1:5]
y_t = df.loc[50:,"lpsa"]

print(x_tr)
model = LinearRegression()
model.fit(x_tr, y_tr)
r2_score = model.score(x_tr, y_tr)
print(f"linear params : {model.coef_}")
print(f"R-squared value: {r2_score}")

n_lambdas = 200
lambdas = np.logspace(-5,5,n_lambdas)

r_coef = []
r_sq = []
for l in lambdas:
    modelRidge = Ridge()
    modelRidge.fit(x_tr,y_tr,l)
    r_coef.append(modelRidge.coef_)
    r_sq.append(modelRidge.score(x_tr,y_tr))

plt.semilogx(lambdas, r_sq)
#plt.show()
# it seems like lambda=10E-5 is the best hyperparameter for Ridge regression

l_coef = []
l_sq = []
for l in lambdas:
    modelLasso = Lasso()
    modelLasso.fit(x_tr,y_tr,l)
    l_coef.append(modelLasso.coef_)
    l_sq.append(modelLasso.score(x_tr,y_tr))

plt.semilogx(lambdas, l_sq)
plt.legend(["Ridge","Lasso"])
plt.xlabel("Lambda")
plt.ylabel("R² coefficient")
plt.show()
#ridge is better for lambda < 8E-5
# lasso has R² constant (=0.02) for all lambdas