import numpy as np


def one_hot(x, k):
    v = np.zeros((k, 1))
    
    v[x-1,:] = 1
    return v
    

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    # Your implementation here
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))
    mistakes = 0
    b = False
    
    while b == False:
        b = True
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[0,i]
            
            if y * (np.dot(th.T, x) + th0) <= 0:
                th = th + y * x
                th0 = th0 + y
                b = False
                mistakes += 1

    print(mistakes)    
    return (th, th0)

def predict(data_test, th, th0):
    print(data_test)
    return np.sign(np.dot(th.T, data_test) + th0) > 0 

def distance_to_separator(data_test, th, th0):
    return (np.dot(th.T, data_test) + th0) / np.sqrt(np.sum(th*th))

initial_data = np.array([[1, 2, 3, 4, 5, 6]])
labels_train = np.array([[1, 1, -1, -1, 1, 1]])
data_train = np.empty((6, 0))

unformat_test = np.array([[1, 6]])
data_test = np.empty((6,0))

for i in initial_data[0]:
    data_train = (np.concatenate((data_train, one_hot(i, 6)),axis=1))
    
for i in unformat_test[0]:
    data_test = np.concatenate((data_test, one_hot(i,6)), axis=1)
    
th, th0 = perceptron(data_train, labels_train)
print(th, th0)
print(predict(data_test, th, th0))
print(distance_to_separator(data_test, th, th0))