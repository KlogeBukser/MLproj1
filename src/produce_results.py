
from plotting import *

import numpy as np
from sklearn import linear_model


def plot_lmb_MSE(lmbs, mses, regression_method, labels):

    '''plot MSE vs lambdas for lasso'''
    plot_2D(lmbs, mses, plot_count = len(mses), title=regression_method+' MSE vs lambdas', x_title='log10(lambdas)', y_title='MSE',label=labels, filename="lasso_MSEvsLMB.pdf", multi_x=False)
