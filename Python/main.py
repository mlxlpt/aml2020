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
from datetime import date

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Product
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C 

#from gmr.utils import check_random_state
#from gmr import MVN, GMM, plot_error_ellipses

#Use diagnostic dataset, change to false for prognostic
USE_NUMPY = True
fileName = 'shalegas' #shalegas, MLO2, noisysine
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
    datanp = np.asarray(data).astype(np.float)
 
    header = list(csv.reader(open(headerpath), delimiter=csvDelimiter))
    header = header[0]
    datapd = pd.DataFrame(data, columns=mangle_dupe_cols(header))
    return datanp, datapd, header

###################
##### Main program
datanp, datapd, headers = load_parse_data()

######## SI TU PREFERES TRAVAILLER AVEC DES TABLEAUX NUMPY, ECRIT data = datanp
if USE_NUMPY:
    data = datanp
    data 
else:
    data = datapd

ratioTrainTest = 0.6
KERNEL_WIDTH = 10
ratioDataset = 1 # percentage of dataset used

x_train,x_test=train_test_split(data,train_size=ratioTrainTest*ratioDataset,\
                                test_size=(1-ratioTrainTest)*ratioDataset,\
                                random_state=int(np.random.rand()*0))

print("\n\nCOLONNES DISPONIBLES:")
print(headers, end='\n\n')

if fileName == 'MLO2' or fileName == 'MLO':
    x_col = ['decimalDate']
    y_col = ['interpolated']
elif fileName == 'noisySine':
    x_col = ['x']
    y_col  = ['y']
elif fileName == 'shalegas':
    #x_col = ['Permian (TX-NM)','Marcellus (PA-WV-OH-NY)']
    #x_col = ['Marcellus (PA-WV-OH-NY)']
    x_col = ['Bakken (ND-MT)','Marcellus (PA-WV-OH-NY)']
    y_col = ['Utica (OH-PA-WV)']

if not USE_NUMPY:
    x_tr = x_train.loc[:,x_col].astype(float)
    y_tr = x_train.loc[:,y_col].astype(float)
    x_te = x_test.loc[:,x_col].astype(float)
    y_te = x_test.loc[:,y_col].astype(float)
else:    
    x_tr = []
    x_te = []
    
    for s in x_col:
        i = 0
        while i < len(headers):
            if s == headers[i]:
                x_tr.append(x_train[:,i])
                x_te.append(x_test[:,i])
                break
            i += 1
    x_tr = np.asarray(x_tr).transpose()
    x_te = np.asarray(x_te).transpose()
    
    i = 0
    for s in y_col:
        i = 0
        while i < len(headers):
            if s == headers[i]:
                y_tr = x_train[:,i]
                y_te = x_test[:,i]
                break
            i += 1
            
#REMOVE POINTS IN NEAR PROXIMITY TO ANOTHER
print(x_tr.shape)

lst = []
for i in range(0,x_tr.shape[0]):
    for j in range(i+1,x_tr.shape[0]):
        nx = np.linalg.norm(x_tr[i]-x_tr[j])
        ny = np.linalg.norm(y_tr[i]-y_tr[j])
        nrm = np.sqrt(nx**2+ny**2)
        if (nrm < 1e-1):
            lst.append(j)
x_tr = np.delete(x_tr, lst, 0)
y_tr = np.delete(y_tr, lst, 0)

lst = []  
for i in range(0,x_te.shape[0]):
    for j in range(i+1,x_te.shape[0]):
        nx = np.linalg.norm(x_te[i]-x_te[j])
        ny = np.linalg.norm(y_te[i]-y_te[j])
        nrm = np.sqrt(nx**2+ny**2)
        if (nrm < 1e-1):
            lst.append(j)          
x_te = np.delete(x_te, lst, 0)
y_te = np.delete(y_te, lst, 0)

i=0
for (xtr, xte) in zip(x_tr.transpose(), x_te.transpose()):
    plt.figure()
    plt.title('x: %s - y: %s' %(x_col[i], y_col[0]))
    plt.scatter(xtr, y_tr, label='train')
    plt.scatter(xte, y_te, label='test')
    plt.show()
    i=i+1
    
print(x_tr.shape)

x_te = np.expand_dims(x_te, -1)
idx=np.argsort(x_te, axis=0)
#order datapoint according to x, with the output.
x_te = x_te[idx.squeeze(-1)]
y_te = y_te[idx.squeeze(-1)]

#np.random.seed(1)
X = np.atleast_2d(x_tr)
y = np.atleast_2d(y_tr).T.ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
for i in range(X.shape[1]):
    if i == 0:
        x_time = np.atleast_2d(np.linspace(int(np.min(X[:,i])-0.5), int(np.max(X[:,i])+0.5), 1000)).T
    else:
        x_time = np.hstack((x_time, np.transpose(np.atleast_2d(np.linspace(int(np.min(X[:,i])-0.5), int(np.max(X[:,i])+0.5), 1000)))))
i=0
#x_time = np.atleast_2d(x_time)

# Instantiate a Gaussian Process model
rbfk = RBF(float(KERNEL_WIDTH))*C(5.0)+RBF(float(KERNEL_WIDTH))*C(5.0)
gp = GaussianProcessRegressor(kernel=rbfk, n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
y_pred_train, sigma = gp.predict(x_time, return_std=True)

i=0
for lx in x_col:
    x_rg = x_time[:,i]
    plt.figure()
    #plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(X[:,i], y, 'r.', markersize=10, label='Train set')
    plt.plot(x_rg, y_pred_train, 'c-', label="Regression")
    
    #plt.plot(x_te, y_te, 'g.', label="Test set")
    #plt.plot(x_te, y_pred, 'b-', label='Test Prediction')
    plt.fill(np.concatenate([x_rg, x_rg[::-1]]),
             np.concatenate([y_pred_train - 1.9600 * sigma,
                            (y_pred_train + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('%s' % lx)
    plt.ylabel('%s' % y_col)
    #plt.ylim(np.min(y)-1, np.max(y)+1)
    #plt.xlim(np.min(X)-0.5, np.max(x_time_future))
    plt.legend(loc='upper left')
    plt.show()
    i=i+1
