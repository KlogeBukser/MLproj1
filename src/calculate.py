import numpy as np


def sq_diff(a,b):
    # Helper function for error models
    return sum([(a[i] - b[i])**2 for i in range(len(a))])

def MSE(y_data,y_model):
    # First param is from the data set. Second param is from model
    n = len(y_data)
    return sq_diff(y_data,y_model)/n

def R2(y_data,y_model):
    # Finds R squared score
    n = len(y_data)
    avg = np.mean(y_data)*np.ones(n)
    return 1 - sq_diff(y_data,y_model)/sq_diff(y_data,avg)



"""
Linear algebra objects
"""
def get_predict(X,beta):
    return np.dot(X,beta)

def find_beta(X,y):
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
