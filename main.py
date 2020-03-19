#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:16:01 2020

@authors: Nathan Müller, Louis Piotet
@course: Advanced Machine Learning - Mini Project GMR vs GPR
"""

import os
import csv

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Use diagnostic dataset, change to false for prognostic
diagnosticSet = False #prognostic is recommended for regression
csvDelimiter = ','

def load_parse_data():
    
    if diagnosticSet:
        fileName='wdbc' #diagnostic
    else:
        fileName='wpbc' #prognostic- Outcome column: N=0, R=1 !
    datapath = os.path.join('dataset', fileName+'_data.csv')
    headerpath = os.path.join('dataset', fileName+'_header.csv')
    
    data = list(csv.reader(open(datapath), delimiter=csvDelimiter))
    header = list(csv.reader(open(headerpath), delimiter=csvDelimiter))
    
    data = pd.DataFrame(data, columns=header[0])
    
    return data, header[0]

###################
##### Main program - Tu peux déplacer ça dans jupyter notebook si tu travailles dessus
data, headers = load_parse_data()

ratioTrainTest = 0.6
ratioDataset = 1 # percentage of dataset used

x_train,x_test=train_test_split(data,train_size=ratioTrainTest*ratioDataset,\
                                test_size=(1-ratioTrainTest)*ratioDataset,\
                                random_state=0)

print(headers) #refer to w*bc_names.txt for details (7. Attribute information)

#tu peux accéder aux colonnes avec leurs en-têtes, 3 cellules donc 3 textures
print("textures of all 3 cells:")
print(data.Texture)

print("\ntrain set, tumor size:")
print(x_train.TumorSize)

