
from plotting import *
from calculate import *
from model import *

import numpy as np
from sklearn import linear_model


def make_container(container_names,n_pol = None):
    """ Makes empty dictionary container

    :container_names: list<String> of names
    :n_pol: int, number of polynomial degrees
    :returns: dictionary

    """
    container_dict = {}
    if n_pol is None:
        for name in container_names:
            container_dict[name] = []
    else:
        for name in container_names:
            container_dict[name] = np.empty(n_pol)

    return container_dict

def find_MSE_Kfold(models, folds):
    n_pol = models[-1].polydeg + 1

    MSEs = np.empty(n_pol)
    for i, model in enumerate(models):
        MSEs[i] = model.cross_validate(folds)

    return MSEs



def plot_MSE_lambda(n,test,lambdas):
    n_pol = test.shape[0]
    poly_degs = np.arange(n_pol)


    plot_2D(np.log10(lambdas), test, plot_count = test.shape[0], label = ['Test ' + str(i) for i in range(n_pol)],
        title='Ridge' + " Test-MSE " + str(n**2) + ' points',x_title="Lambda",y_title="Error",filename= 'Ridge' + ' BiVa_boot.pdf', multi_x=False)


def plot_ridge(n,models,x_test,z_test,n_boots):
    z_train = models[0].z_train

    n_pol = len(models)
    lambdas = ['best', 0, 0.001, 0.01, 0.1]
    for lamb in lambdas:
        model.set_lambda(0)

    pred,fit = make_predictions_boot(models,x_test,n_boots)
    MSE_test = find_boot_MSE(z_test, pred)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)
    MSE_comp['ridge'] = MSE_test

    for model in models:
        model.set_lambda(0)

    pred,fit = make_predictions_boot(models,x_test,n_boots)
    MSE_boot = find_MSE_boot(z_train, z_test, pred, fit)
    MSE_comp['ols'] = MSE_boot['test']

    plot_2D(range(n_pol), [MSE_comp['ridge'],MSE_comp['ols']], plot_count = 2, label = ['Ridge','OLS'],
        title='Ridge' + " OLS comparison " + str(n**2) + ' points',x_title="Lambda",y_title="Error",filename= 'Ridge' + ' BiVa_boot.pdf', multi_x=False)

    plot_2D(range(n_pol), [MSE_comp['ridge'],MSE_comp['ols'],MSE_Kfold], plot_count = 3, label = ['Test error','Training error', 'K-fold predictions'],
        title='Ridge' + " Kfold prediction for test error " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= regression_method + ' Kfold_test.pdf', multi_x=False)




def plot_lmb_MSE(lmbs, mses, regression_method, labels):

    '''plot MSE vs lambdas for ridge and lasso'''
    plot_2D(lmbs, mses, plot_count = len(mses), title=regression_method+' MSE vs lambdas', x_title='log10(lambdas)', y_title='MSE',label=labels, multi_x=False)
