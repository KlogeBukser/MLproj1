
from plotting import *

import numpy as np
from sklearn import linear_model




def plot_beta(n,betas, file_dir):
    """ Plots beta values for OLS method """

    beta_ranges = [np.arange(len(beta)) for beta in betas]
    plot_2D(beta_ranges, betas, plot_count = len(betas), title='ols beta ' + str(n**2) + ' points',x_title='features',y_title='Beta',filename= 'ols beta.pdf', file_dir=file_dir)


def plot_simple_scores(n,MSE_vals,R2_vals, file_dir):

    poly_degs = np.arange(len(MSE_vals))
    # Plots R2 score over polynomial degrees
    plot_2D(poly_degs, R2_vals, title='ols R$^2$ Score ' + str(n**2) + ' points',x_title='polynomial degree',y_title='R2',filename= 'ols' + ' R2.pdf',file_dir=file_dir)

    # Plots MSE score over polynomial degrees
    plot_2D(poly_degs, MSE_vals,title='ols Mean Squared Error ' + str(n**2) + ' points', x_title='polynomial degree', y_title='MSE', filename='ols' + ' MSE.pdf',file_dir=file_dir)


def plot_boot_scores(n,poly_degs,test_score,train_score,bias,var,Kfold_score, file_dir):

    plot_2D(poly_degs, [test_score,bias,var], plot_count = 3, label = ['MSE','bias','variance'],
        title='ols Bias-Variance ' + str(n**2) + ' points',x_title='polynomial degree',y_title='Error',filename= 'ols BiVa_boot.pdf', multi_x=False,file_dir=file_dir)

    plot_2D(poly_degs, [test_score,train_score], plot_count = 2, label = ['test','train'],
        title='ols MSE comparison ' + str(n**2) + ' points',x_title='polynomial degree',y_title='MSE',filename= 'ols MSE_comp.pdf', multi_x=False,file_dir=file_dir)

    plot_2D(poly_degs, [test_score,Kfold_score], plot_count = 2, label = ['Test error', 'K-fold predictions'],
        title='ols Kfold prediction for test error ' + str(n**2) + ' points',x_title='polynomial degree',y_title='Error',filename= 'ols Kfold_test.pdf', multi_x=False,file_dir=file_dir)



def plot_lmb_MSE(lmbs, mses, regression_method, labels, filename, file_dir):

    '''plot MSE vs lambdas for lasso'''
    plot_2D(lmbs, mses, plot_count = len(mses), title=regression_method+' MSE vs lambdas', x_title='log10(lambdas)', y_title='MSE',label=labels, filename=filename, multi_x=False,file_dir=file_dir)
