# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:20:08 2022

@author: Ivan
"""

import numpy as np

def score(data, labels, ths, th0s):
    pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s))
    print(pos)
    return np.sum(pos == labels, axis=1, keepdims=True)

data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
labels = np.array([-1, -1, +1, +1, +1])
th = np.array([[1, 2, 3], [1, 1, 3]])
th0 = np.array([[-2, 2, 2]])

print(th0*2-1)