import numpy as np


def MSE(y,y_,scale = True):
    n = len(y)
    square = sum([(y[i] - y_[i])**2 for i in range(n)])
    if (scale):
        return square/n
    return square


def R2(y,y_):
    n = len(y)
    avg = np.ones(n)*np.mean(y)
    return 1 - MSE(y,y_,False)/MSE(y,avg,False)



"""
Linear algebra objects
"""
def get_model(X,beta):
    return np.dot(X,beta)

def find_coeffs(X,y):
    # Finds beta
    square = np.dot(X.T,X)                      #nf*nf
    if (np.linalg.det(square) != 0):
        inv = np.linalg.inv(square)             #nf*nf
        beta = np.dot(np.dot(inv,X.T),y)        #nf*1
        return beta

    else:
        print("You dun goofed")


def calc_design(x,functions):

    n = x.shape[0]
    n_funcs = len(functions)
    design = np.ones((n, n_funcs))

    for i in range(n):
        for j in range(n_funcs):
            design[i,j] = functions[j](x[i])

    return design
