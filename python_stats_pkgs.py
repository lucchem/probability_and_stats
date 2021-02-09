#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

import scipy.stats as sts

import sklearn.model_selection as mdl
import sklearn.linear_model as lm
import sklearn.datasets as dat
import sklearn.preprocessing as prp

import statistics
import random

import matplotlib.pyplot as plt
# In[ ]:

# # Python Standard Library
# 
# ## Random number generation with ``random``
# 
# Random integers are generated via Mersenne twister.
# 
# The seeder initializes the random generator state to something, and you can see that ``getstate()`` changes when ``seed()`` is fed something different.
# 
# Many distributions available.
# 
# ## ``statistics``
# 
# Functions to compute statistics of lists. Interesting ``NormalDist`` object for manipulating and sampling normal variates.


# throw a seed
random.seed(a=10102)

# random number between 1 and 10
print(random.randint(1,10))
print()

# unsorted sample without replacement and some random numbers
print(random.sample(range(1,11),3),random.uniform(1,2),random.triangular(1,3,mode= 1.5),random.expovariate(2))
print()

# list of Gaussian variates
lista =[ random.gauss(mu = 0, sigma = 1) for i in range(1,10) ]
print(lista,statistics.mean(lista),statistics.pstdev(lista))
print()

# Gaussian distributions
d1 = statistics.NormalDist(mu = 1, sigma = 1)
d2 = statistics.NormalDist(mu = 2, sigma = 1)
print(d1.overlap(d2))
d3 = d1 + d2
print(d3.mean, d3.variance, d3.samples(3))
d1 = 5*d1+3
print(d1)

# In[ ]:

# # NumPy and SciPy
# 
# ## Random numbers and statistics with ``numpy``
# 
# It contains a bit of everything and stores stuff in arrays, whose methods are used for doing stats.
# 
# See: https://numpy.org/doc/1.19/reference/random/index.html
# 
# Basically, it contains a class called ``Generator`` that interfaces the user to everything. The ``random`` module
# 
# > .. contains pseudo-random number generator with a number of methods that are similar to the ones available in
# Generator. It uses Mersenne Twister, and this bit generator can be accessed using MT19937.
# Generator, besides being NumPy-aware, has the advantage that it provides a much larger number of probability distributions to choose from.


# instantiate an object of class Generator
generator = np.random.default_rng(10101)

# 7 sequences of 3 fair coin flips
flips = generator.binomial(n = 3, p = 0.5, size = 7)
print(flips,end='\n\n')

# some random variates
print(generator.uniform(), generator.poisson(lam = 4), generator.normal(loc = 2, scale = 2))

# sample a chi-square distribution with 7 degrees of freedom
chisq = generator.chisquare(df = 7, size = 11)
print(chisq,end='\n\n')

# play around with permutations and combinations
permuted = generator.permutation(range(10))
combinat = generator.choice(range(10), size = 4, replace = False, shuffle = False)
print(combinat)
print(permuted,end='\n\n')

# some stats
sample1 = generator.normal(loc = 2, scale = 2, size = 100)
sample2 = 2*sample1+1
print(sample1.mean(), sample1.var(ddof = 1), np.quantile(sample1, 0.7))
print(np.corrcoef(sample1, sample2))
print('regression line slope: ',np.corrcoef(sample1, sample2)[0][1]*sample2.std(ddof = 1)/sample1.std(ddof = 1))

# In[ ]:

# ## ``Scipy.stats``
# 
# Even more capable than the others, see https://docs.scipy.org/doc/scipy/reference/stats.html.
# 
# It includes more correlation models and many statistical tests, and somehow is complementary to ``numpy.random`` 
# and NumPy arrays.


# some stats
sample1 = generator.exponential(scale = 2, size = 100)
sample2 = 2*sample1**2+1
print(sts.mstats.spearmanr(sample1,sample2),sts.mstats.spearmanr(sample1,sample2).correlation)
print(sts.mstats.kendalltau(sample1,sample2))

# try to mess up with sample2
sample1 = generator.exponential(scale = 2, size = 100)
sample2 = sample1*generator.normal(size = len(sample1))*generator.exponential(scale = 1, size = len(sample1))
print(sts.mstats.spearmanr(sample1,sample2))
print(sts.mstats.kendalltau(sample1,sample2))

# In[ ]

# Pandas

# ``Series`` and ``DataFrame`` are classes able to contain and process data, also statistical ones.
# Data can be passed as NumPy arrays or just lists.
# Examples taken from Pandas guide

series = pd.Series([1, 3, 5, np.nan, 6, 8], name = 'values 2B statsed')
print(series)
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
df.reset_index(inplace = True)
df['ciao'] = series
print(df)

df2 = pd.DataFrame({'A': 1.,
'B': pd.date_range('20130101', periods=4),
'C': pd.Series(1, index=list(range(4)), dtype='float32'),
'D': np.array([3] * 4, dtype='int32'),
'E': pd.Categorical(["test", "train", "test", "train"]),
'F': 'foo'})

df2.set_index('B',inplace = True)
print(df2)

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))

# In[ ]

# scikit-learn
#
# Can load datasets and linear models to regress stuff

dataset = dat.load_boston(return_X_y = True)
X = pd.DataFrame()
X['1'] = dataset[0][:,5]
X['2'] = np.log(dataset[0][:,12])
y = dataset[1]

X = (X-X.mean())/X.std(ddof = 1)
y = (y-y.mean())/y.std(ddof = 1)

# In[]

# regression with scikit-learn

X = generator.exponential(scale = 2, size = 1000)
error = generator.exponential(scale = 2, size = 1000) - 0.5
#error = generator.normal(loc = 0, scale = 10, size = 1000)
y = error + 2*X +1*X**2

X = X.reshape(-1,1)
y = y.reshape(-1,1)

scaler = prp.StandardScaler()

X_scal = X#scaler.fit_transform(X)
y_scal = y#scaler.fit_transform(y)

X_train, X_test, y_train, y_test = mdl.train_test_split(X_scal, y_scal, test_size=0.25, random_state=10101)

reg = lm.LinearRegression()
reg.fit(X_train, y_train)
beta = reg.coef_[0][0]
q = reg.intercept_[0]
residuals = y_test - X_test*beta

print(reg.score(X_test,y_test),beta,q)

plt.scatter(X_train,y_train)
plt.scatter(X_test, X_test*beta+q)
plt.show()

sts.probplot(residuals.reshape(1,250)[0], dist="norm", plot = plt)