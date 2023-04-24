# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 04:39:17 2022

@author: Ivan
"""

import numpy as np

def margin(x, y, th, th0):
    return y*(np.dot(th.T, x) + th0) / np.sqrt(np.sum(th*th))

def hinge_loss(x, y, th, th0, ref):
    marg = margin(x, y, th, th0)
    cond = np.where(marg < ref, marg, 0*marg) / ref
    
    return np.where(cond == 0, cond, 1-cond)
  

def test_separator(data, labels, th, th0):
    a = margin(data, labels, th, th0)
    print(a)
    return (np.sum(a),
            np.min(a),
            np.max(a))
    

data = np.array([[1, 4, 3],
                 [1, 1, 2]])
labels = np.array([[-1, -1, 1]])
thA = np.array([[-0.0737901, 2.40847205]]).T
thA0 = -3.492621154916483
thB = np.array([[-0.23069578, 2.55735501]]).T
thB0 = -3.3857770692522666
ref = 1/2.56773931506

print(np.sum(hinge_loss(data, labels, thB, thB0, ref))/3)



