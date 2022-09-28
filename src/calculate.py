import numpy as np


def sq_diff(a,b):
    # Helper function for error models
    return sum([(a[i] - b[i])**2 for i in range(len(a))])

# def MSE(y_data,y_model):
    # # First param is from the data set. Second param is from model
    # n = len(y_data)
    # return sq_diff(y_data,y_model)/n


def R2(y_data,y_model):
    # Finds R squared score
    n = len(y_data)
    avg = np.mean(y_data)*np.ones(n)
    return 1 - sq_diff(y_data,y_model)/sq_diff(y_data,avg)




# some of these are from lecture notes with modification
def MSE(y_test, y_pred):
    """computes the mean squared error for a given prediction

    :y_test: TODO
    :y_pred: TODO
    :returns: TODO

    """
    return np.mean( np.mean((y_test - y_pred)**2, keepdims=True) )


def cal_bias(y_test, y_pred):
    """computes the bias for a given prediction

    :y_test: array like
    :y_pred: array like
    :returns: bias

    """
    return np.mean( (y_test - np.mean(y_pred, keepdims=True))**2 )

def cal_variance(y_test, y_pred):
    """computes the variance for a given prediction

    :y_test: array-like
    :y_pred: array-like
    :returns: variance

    """
    return np.mean( np.var(y_pred, keepdims=True) )
    



