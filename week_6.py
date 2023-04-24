# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 02:34:16 2022

@author: Ivan
"""

import numpy as np

def SM(z):
    d, n = z.shape
    a = np.zeros((d, n))
    
    denom = 0
    
    for i in range(n):
        denom += np.e**z[:,i:i+1]
    
    for i in range(n):
        a[:,i:i+1] = np.e**z[:,i:i+1] / denom
    
    return a

def NLL(a, y):
    return -np.sum(y*np.log(a))

def NLL_grad(x, a, y):
    return x * (a - y)

def asd(x, y, w, w0):
    d, n = w.shape
    z = np.dot(x.T, w)
    a = SM(z)
    
    print(a)
    
    nll_grad = np.dot(x,(a - y.T))
    
    w1 = w - 0.5*nll_grad
    
    z = np.dot(x.T, w1)
    a = SM(z)
    
    print(a)

        

w = np.array([[1, -1, -2], [-1, 2, 1]])
w0 = np.array([[0, 0, 0]])

x = np.array([[1, 1]]).T
y = np.array([[0, 1, 0]]).T

asd(x, y, w, w0)