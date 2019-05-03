#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:19:20 2019

@author: michaelberry
"""

#load toolboxes

import os
from sys import argv
from os import mkdir,path,makedirs
import numpy as np
import scipy as sc
from scipy import linalg, stats, io
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import argparse
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import re
import math

data_file="test.txt"


def load_data():
    numFiles = 20
    N   = 301
    M   = 201
    cnt = 0
    
    X   = []  # Data Array   
    Y   = []  # Label Array
       
    for k in range(0, numFiles):
        
        fileID = "%02d"%(k)
        data_Dir = '/home/michaelberry/Research/ml_work/Data/DataModel'+sr+'/topos'+fileID+'/'
        files = [f for f in os.listdir(data_Dir) if f.endswith(suffs)] #make list of files in directry
        
        for f in files:

            label_perf  = np.zeros(1)
            match       = re.search('(?<=kb).*(?=_run)', f) #use RegEx to pull label directly from the file name 
            label       = match.group(0) if match else None
            labelp[0]   = float(label)
            dataFile    = data_Dir + f                      #string to file the file with
            data        = pd.read_csv(dataFile, delimiter="\t", header=None) #read file
            data1D      = data[0][:]                        #landscape to 1D np array
            # find Slope - K
            surf        = np.resize(data,(N,M))
            S           = slope_funct(surf)                 #determine slope value for landscape
            
            nD_label   = float(label_perf * S)
            #build the matrix
            X[cnt][:]  = data1D
            Y[cnt]     = label_perf
            Yn[cnt]    = nD_label

            cnt += 1
            print(cnt)
            
    #remove excess sections of arrays
    X  = np.delete(X, slice(cnt,30000), axis=0)
    Y  = np.delete(Y, slice(cnt,30000), axis=0)
    Yn  = np.delete(Yn, slice(cnt,30000), axis=0)
    
    return [X, Y, Yn]


def slope_funct(arr):
    N = 301
    M = 201
#    #determines the median slope value for a surface
#    # Padded copy of arr
    B=np.empty((N+2,M+2))
    B[1:-1,1:-1]=arr
    B[0,1:-1]=arr[0,:]
    B[-1,1:-1]=arr[-1,:]
    B[1:-1,0]=arr[:,0]
    B[1:-1,-1]=arr[:,-1]
    B[0,0]=arr[1,1]
    B[-1,-1]=arr[-1,-1]
    B[-1,0]=arr[-1,0]
    B[0,1]=arr[0,1]

    # Compute 4 absolute differences
    D1=np.abs(B[1:,1:-1]-B[:-1,1:-1]) # first dimension
    D2=np.abs(B[1:-1,1:]-B[1:-1,:-1]) # second dimension
    D3=np.abs(B[1:,1:]-B[:-1,:-1]) # Diagonal
    D4=np.abs(B[1:,:-1]-B[:-1,1:]) # Antidiagonal

    # Compute maxima in each direction
    M1=np.maximum(D1[1:,:],D1[:-1,:])/200
    M2=np.maximum(D2[:,1:],D2[:,:-1])/200
    M3=np.maximum(D3[1:,1:],D3[:-1,:-1])/(200*math.sqrt(2))
    M4=np.maximum(D4[1:,:-1],D4[:-1,1:])/(200*math.sqrt(2))

    M=np.max(np.dstack([M1,M2,M3,M4]),axis=2)

    slopes=M.reshape([1,-1])
    # clear things 
    del B
    del D1
    del D2
    del D3
    del D4
    del M1
    del M2
    del M3
    del M4
    del M
    return np.median(slopes)


if __name__ == '__main__':
    
    X, Y, Yn = load_data()
    
    X = MinMaxScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    # KRR
    print("KRR")
    model = KernelRidge(alpha=3e-1)  # alpha=5.0e0, alpha=1.0e2l, alpha=5e-2
    
    y_pred = model.fit(X_train,y_train).predict(X_test)
    print("done")
    
#    print diff, diff1
    print(np.median(np.abs(y_pred - y_test)/y_test))

    print(metrics.median_absolute_error(y_pred, y_test)/np.median(y_test))
    print(metrics.median_absolute_error(y_pred[0:10], y_test[0:10])/np.median(y_test[0:10]))
    
    file = open(data_file,"w+")
    import csv
    with open(data_file,'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(y_test,y_pred))#y_test_nD,y_pred_nD))
