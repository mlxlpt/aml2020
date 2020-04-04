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
from pandas import Index

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Product
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C 

from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses

#Use diagnostic dataset, change to false for prognostic
prognosticSet = False #prognostic is recommended for regression

fileName = 'ozonedepletor' #MLO, MLO2 or ozonedepletor (je préfère le depletor perso)
csvDelimiter = ','

#renommer les colonnes portant le même nom que d'autres
def mangle_dupe_cols(columns):
    counts = {}
    for i, col in enumerate(columns):
        cur_count = counts.get(col, 0)
        if cur_count > 0:
            columns[i] = '%s.%d' % (col, cur_count)
        counts[col] = cur_count + 1
    return columns

def load_parse_data():
    datapath = os.path.join('../dataset', fileName+'_data.csv')
    headerpath = os.path.join('../dataset', fileName+'_header.csv')
    
    data = list(csv.reader(open(datapath), delimiter=csvDelimiter))

    header = list(csv.reader(open(headerpath), delimiter=csvDelimiter))
    header = header[0]
    data = pd.DataFrame(data, columns=mangle_dupe_cols(header))
    return data, header

###################
##### Main program
data, headers = load_parse_data()

ratioTrainTest = 0.6
KERNEL_WIDTH = 10.0

ratioDataset = 1 # percentage of dataset used

x_train,x_test=train_test_split(data,train_size=ratioTrainTest*ratioDataset,\
                                test_size=(1-ratioTrainTest)*ratioDataset,\
                                random_state=int(np.random.rand()*0))

print("\n\nCOLONNES DISPONIBLES:")
print(headers, end='\n\n')

if fileName == 'MLO2' or fileName == 'MLO':
    x_col = 'decimalDate'
    y_col = 'interpolated'
elif fileName == 'ozonedepletor':
    x_col = 'date'
    y_col  = 'CH3Br'
    #y_col = 'HFC-152a'

x_tr = x_train.loc[:,x_col].astype(float)
y_tr = x_train.loc[:,y_col].astype(float)
x_te = x_test.loc[:,x_col].astype(float)
y_te = x_test.loc[:,y_col].astype(float)

#idtr=y_tr.isna()
#idte=y_te.isna()
#y_tr = y_tr[np.invert(idtr.to_numpy())]
#y_te = y_te[np.invert(idte.to_numpy())]
#x_tr = x_tr[np.invert(idtr.to_numpy())]
#x_te = x_te[np.invert(idte.to_numpy())]

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
rbfk = RBF(float(KERNEL_WIDTH))*C(10.0)
gp = GaussianProcessRegressor(kernel=rbfk, n_restarts_optimizer=50)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
x_time_future = np.atleast_2d(np.linspace(int(np.max(X)-5), int(np.max(X)+20), 1000)).T
y_pred, sigma = gp.predict(x_time_future, return_std=True)
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
y_pred_train, sigma = gp.predict(x_time, return_std=True)
plt.figure()
#plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Train set')
plt.plot(x_time, y_pred_train, 'c-', label="Regression")
#plt.plot(x_te, y_te, 'g.', label="Test set")
plt.plot(x_time_future, y_pred, 'b-', label='Future Prediction')
#plt.plot(x_te, y_pred, 'b-', label='Test Prediction')
plt.fill(np.concatenate([x_time, x_time[::-1]]),
         np.concatenate([y_pred_train - 1.9600 * sigma,
                        (y_pred_train + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$MGSMR$')
#plt.ylim(np.min(y)-1, np.max(y)+1)
#plt.xlim(np.min(X)-0.5, np.max(x_time_future))
plt.legend(loc='upper left')
