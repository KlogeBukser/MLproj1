import numpy as np


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
