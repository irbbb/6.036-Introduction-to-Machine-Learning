# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:36:18 2022

@author: Ivan
"""

import numpy as np

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])
        
        
def gd(f, df, x0, step_size_fn, max_iter):
    x = x0.copy()
    fs = [f(x)]
    xs = [x]
    i = 0
    
    while i < max_iter:
        i = i + 1
        x = x - step_size_fn(i) * df(x)
        xs.append(x)
        fs.append(f(x)) 
    
    return (x, fs, xs)
    
def num_grad(f, delta = 0.001):
    def df(x):
        d, n = x.shape
        res = np.empty((0,1))
        
        for i in range(d):
            delta_v = np.zeros((d, n))
            delta_v[i,:] = delta
            res_f = (f(x + delta_v) - f(x - delta_v)) / (2 * delta)
            res_f = np.array([res_f])
            res = np.vstack((res, res_f))
            
        return res
        
    return df
    

def minimize(f, x0, step_size_fn, max_iter):
    x = x0.copy()
    fs = [f(x)]
    xs = [x]
    df = num_grad(f)
    
    for i in range(max_iter):
        x = x - step_size_fn(1) * df(x)
        xs.append(x)
        fs.append(f(x))
        
    return (x, fs, xs)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


def hinge(v):
    v = 1 - v
    h_loss = np.where(v >= 0, v, 0)
    return h_loss
    

def hinge_loss(x, y, th, th0):
    v = y * (np.dot(th.T, x) + th0)
    return hinge(v)

def svm_obj(x, y, th, th0, lam):
    d, n = x.shape
    h_loss = hinge_loss(x, y, th, th0)
    return 1/n * np.sum(h_loss) + lam*(np.sum(th*th))


def d_hinge(v):
    # Revisar si 1 - y derivada
    h_loss = np.where(v >= 1, 0, -1)
    return h_loss

def d_hinge_loss_th(x, y, th, th0):
    v = y*(np.dot(th.T, x) + th0)
    return d_hinge(v) * (y*x)

def d_hinge_loss_th0(x, y, th, th0):
    v = y*(np.dot(th.T, x) + th0)
    
    return d_hinge(v) * y

def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis = 1, keepdims = True) + lam * 2 * th

def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

# Return gradient of theta
def svm_obj_grad(X, y, th, th0, lam):
    grad_th = d_svm_obj_th(X, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))
    
    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)

    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)
    
    return gd(f, df, init, svm_min_step_size_fn, 10)  

# Test cases


def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

x_1, y_1 = super_simple_separable()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))
# print(ans)

x_1, y_1 = separable_medium()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))
print(ans)

# X1 = np.array([[1, 2, 3, 9, 10]])
# y1 = np.array([[1, 1, 1, -1, -1]])
# th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
# X2 = np.array([[2, 3, 9, 12],
#                [5, 2, 6, 5]])
# y2 = np.array([[1, -1, 1, -1]])
# th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

# d_hinge(np.array([[ 71.]])).tolist()
# d_hinge(np.array([[ -23.]])).tolist()
# print(d_hinge(np.array([[ 71, -23.]])).tolist())

# d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
# print(d_hinge_loss_th(X2, y2, th2, th20).tolist())
# d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
# print(d_hinge_loss_th0(X2, y2, th2, th20).tolist())

# print(d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())
# print(d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist())
# d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
# print(d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist())

# svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
# print(svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())