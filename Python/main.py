#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:16:01 2020

@authors: Nathan Müller, Louis Piotet
@course: Advanced Machine Learning - Mini Project GMR vs GPR
"""

#reset environment - comme clear all sur matlab :')
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
import csv

import numpy as np
import pandas as pd

from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C 

from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses

#Use diagnostic dataset, change to false for prognostic
prognosticSet = False #prognostic is recommended for regression
csvDelimiter = ','

def mangle_dupe_cols(columns):
    counts = {}
    for i, col in enumerate(columns):
        cur_count = counts.get(col, 0)
        if cur_count > 0:
            columns[i] = '%s.%d' % (col, cur_count)
        counts[col] = cur_count + 1
    return columns

def load_parse_data():
    
    if not prognosticSet:
        fileName='MLO' #diagnostic
    else:
        fileName='wpbc' #prognostic- Outcome column: N=0, R=1 !
    datapath = os.path.join('../dataset', fileName+'_data.csv')
    headerpath = os.path.join('../dataset', fileName+'_header.csv')
    
    data = list(csv.reader(open(datapath), delimiter=csvDelimiter))
    header = list(csv.reader(open(headerpath), delimiter=csvDelimiter))
    header = header[0]
    data = pd.DataFrame(data, columns=mangle_dupe_cols(header))
    return data, header

###################
##### Main program - Tu peux déplacer ça dans jupyter notebook si tu travailles dessus
data, headers = load_parse_data()

ratioTrainTest = 0.7
ratioDataset = 1 # percentage of dataset used

x_train,x_test=train_test_split(data,train_size=ratioTrainTest*ratioDataset,\
                                test_size=(1-ratioTrainTest)*ratioDataset,\
                                random_state=int(np.random.rand()*10))

#print(headers) #refer to w*bc_names.txt for details (7. Attribute information)


x_col = 'decimalDate'
y_col = 'interpolated'

x_tr = x_train.loc[:,x_col].astype(float)
y_tr = x_train.loc[:,y_col].astype(float)
x_te = x_test.loc[:,x_col].astype(float)
y_te = x_test.loc[:,y_col].astype(float)

x_te = np.expand_dims(x_te, -1)
idx=np.argsort(x_te, axis=0)
#order datapoint according to x, with the output.
x_te=x_te[idx.squeeze(-1)]
y_te = y_te[idx.squeeze(-1)]

#np.random.seed(1)
X = np.atleast_2d(x_tr).T
y = np.atleast_2d(y_tr).T.ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x_time = np.atleast_2d(np.linspace(int(np.min(X)-0.5), int(np.max(X)+0.5), 1000)).T

# Instantiate a Gaussian Process model
kernel = RBF(1, (1e-5, 1e5))*C(10.0, (1e-4, 1e4)) 
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x_te, return_std=True)
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
y_pred_train, sigma = gp.predict(x_time, return_std=True)
plt.figure()
#plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Train set')
plt.plot(x_time, y_pred_train, 'c-', label="Train Prediction")
#plt.plot(x_te, y_te, 'g.', label="Test set")
#plt.plot(x_te, y_pred, 'b-', label='Test Prediction')
plt.fill(np.concatenate([x_time, x_time[::-1]]),
         np.concatenate([y_pred_train - 1.9600 * sigma,
                        (y_pred_train + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(np.min(y)-1, np.max(y)+1)
plt.xlim(np.min(X)-0.5, np.max(X)+0.5)
plt.legend(loc='upper left')
