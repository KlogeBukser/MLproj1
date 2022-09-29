import numpy as np

def R2(z_test,z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: R squared score

    """
    n = len(z_test)
    avg = np.mean(z_test)*np.ones(n)

    return 1 - np.mean((z_test - z_pred)**2)/np.mean((z_test - avg)**2)


# some of these are from lecture notes with modification
def MSE(z_test, z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: Mean squared error

    """
    return np.mean( np.mean((z_test - z_pred)**2, keepdims=True) )

def MSE2(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n



def cal_bias(z_test, z_pred):
    """computes the bias for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: bias

    """
    return np.mean( (z_test - np.mean(z_pred, keepdims=True))**2 )

def cal_variance(z_test, z_pred):
    """computes the variance for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: variance

    """
    return np.mean( np.var(z_pred, keepdims=True) )
