#!/usr/bin/env python3
import numpy as np
import pandas as pd

#import sklearn.preprocessing as prp
import sklearn.linear_model as lm
import sklearn.model_selection as ms

import scipy.stats as sts

import matplotlib.pyplot as plt

#%%

N = 1000

generator = np.random.default_rng(10101)

X = pd.DataFrame()

X["1"] = generator.exponential(scale = 0.5, size = N)
X["2"] = generator.normal(loc = 5, scale = 2, size = N)
X["3"] = 0.001*X["2"] + 0.1*np.cos(X["1"])# + generator.normal(scale = 0.01, size = N)

print(X.corr())

#%%

error = generator.standard_t(df = 4, size = N)
#error = generator.standard_normal(size = N)

coefficients = np.array([2.5, 1.2, -3])

y = X @ coefficients + error

sts.probplot(error, plot = plt)
plt.title('Error QQ-plot')
plt.show()

plt.scatter(X["1"],y, label = "1")
plt.scatter(X["2"],y, label = "2")
plt.scatter(X["3"],y, label = "3")
plt.legend()
plt.show()
#%%

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size = 0.8, random_state = 10101)

model = lm.LinearRegression()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)

print(model.intercept_,model.coef_)
print(model.score(X_train,y_train), model.score(X_test,y_test))

residuals = y_test-model.intercept_-X_test @ model.coef_

sts.probplot(residuals.values, plot = plt)
plt.title('Residuals QQ-plot: normal?')
plt.show()

plt.scatter(y_test,residuals)
plt.title('residuals v test data: homoskedastic?')
plt.show()

plt.scatter(X_test["1"],y_test, label = "1")
plt.scatter(X_test["2"],y_test, label = "2")
plt.scatter(X_test["3"],y_test, label = "3")
plt.scatter(X_test["1"],y_hat, label = "1")
plt.scatter(X_test["2"],y_hat, label = "2")
plt.scatter(X_test["3"],y_hat, label = "3")
plt.title('Prediction vs test: goodness-of-fit : R2={:.4f}'.format(model.score(X_test,y_test)))
plt.show()

#%%
ridge = lm.RidgeCV(alphas=[0.01*i for i in range(1,200) ])
ridge.fit(X_train, y_train)

y_hat = ridge.predict(X_test)

print(ridge.intercept_,ridge.coef_)
print(ridge.score(X_train,y_train), ridge.score(X_test,y_test))

residuals = y_test-ridge.intercept_-X_test @ ridge.coef_

sts.probplot(residuals.values, plot = plt)
plt.title('Ridge residuals QQ-plot: normal?')
plt.show()

plt.scatter(y_test,residuals)
plt.title('Ridge residuals v test data: homoskedastic?')
plt.show()

plt.scatter(X_test["1"],y_test, label = "1")
plt.scatter(X_test["2"],y_test, label = "2")
plt.scatter(X_test["3"],y_test, label = "3")
plt.scatter(X_test["1"],y_hat, label = "1")
plt.scatter(X_test["2"],y_hat, label = "2")
plt.scatter(X_test["3"],y_hat, label = "3")
plt.title('Ridge prediction vs test: goodness-of-fit : R2={:.4f}'.format(ridge.score(X_test,y_test)))
plt.show()

#%%
lasso = lm.LassoCV()
lasso.fit(X_train, y_train)

y_hat = lasso.predict(X_test)

print(lasso.intercept_,lasso.coef_)
print(lasso.score(X_train,y_train), lasso.score(X_test,y_test))

residuals = y_test-lasso.intercept_-X_test @ lasso.coef_

sts.probplot(residuals.values, plot = plt)
plt.title('LASSO residuals QQ-plot: normal?')
plt.show()

plt.scatter(y_test,residuals)
plt.title('LASSO residuals v test data: homoskedastic?')
plt.show()

plt.scatter(X_test["1"],y_test, label = "1")
plt.scatter(X_test["2"],y_test, label = "2")
plt.scatter(X_test["3"],y_test, label = "3")
plt.scatter(X_test["1"],y_hat, label = "1")
plt.scatter(X_test["2"],y_hat, label = "2")
plt.scatter(X_test["3"],y_hat, label = "3")
plt.title('LASSO prediction vs test: goodness-of-fit : R2={:.4f}'.format(lasso.score(X_test,y_test)))
plt.show()