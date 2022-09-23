import numpy as np


def sq_diff(a,b):
    return sum([(a[i] - b[i])**2 for i in range(len(a))])

def MSE(y,y_):
    n = len(y)
    return sq_diff(y,y_)/n

def R2(y,y_):
    n = len(y)
    avg = np.ones(n)*np.mean(y)
    return 1 - sq_diff(y,y_)/sq_diff(y,avg)



"""
Linear algebra objects
"""
def get_model(X,y):

    beta = find_coeffs(X,y)
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


def calc_design(vars,functions):

    n = vars.shape[0]
    n_funcs = len(functions)
    design = np.ones((n, n_funcs))

    for i in range(n):
        for j in range(n_funcs):
            design[i,j] = functions[j](vars[i])

    return design
