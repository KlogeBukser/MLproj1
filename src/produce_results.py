
from plotting import *
from calculate import *
from model import *

import numpy as np
from sklearn import linear_model

ALLOWED_METHODS = ['ols','ridge','lasso']

ERR_INVALID_METHOD = 'Invalid regression_method, allowed methods are {0}, {1}, {2}'
ERR_INVALID_METHOD.format(*ALLOWED_METHODS)


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


def make_predictions_boot(models, x_test, n_boots = 100):
    """ Makes predictions on models with bootstrap resampling

    :models: array_like of Model objects
    :x_test: array_like of tuples
    :n_boots: int, number of bootstraps
    :returns: 2D array_like, 2D array_like

    """
    n_pol = len(models)
    n_test = len(x_test)
    n_train = models[0].X_train.shape[0]
    poly_degs = np.arange(n_pol)
    z_pred = np.empty((n_pol, n_test, n_boots))
    z_fit = np.empty((n_pol, n_train, n_boots))

    for i, model in enumerate(models):
        z_pred[i], z_fit[i] = model.bootstrap(x_test,n_boots)
    return z_pred, z_fit


def find_MSE_boot(z_train,z_test,z_pred,z_fit):
    """ Plots MSE score for models against their polynomial degree on test set and training set.
    For bootstrap predictions


    :z_test: array_like
    :z_pred: 2D array_like

    """

    n_pol = z_pred.shape[0]
    MSE_dict = make_container(['test','train','bias','variance'],n_pol)
    for i in range(n_pol):

        MSE_dict['test'][i] = MSE(z_test, z_pred[i])
        MSE_dict['train'][i] = MSE(z_train,z_fit[i])
        MSE_dict['bias'][i] = cal_bias(z_test,z_pred[i])
        MSE_dict['variance'][i] = cal_variance(z_pred[i])

    return MSE_dict


def plot_MSE_lambda(n,test,lambdas):
    n_pol = test.shape[0]
    poly_degs = np.arange(n_pol)


    plot_2D(np.log10(lambdas), test, plot_count = test.shape[0], label = ['Test ' + str(i) for i in range(n_pol)],
        title='Ridge' + " Test-MSE " + str(n**2) + ' points',x_title="Lambda",y_title="Error",filename= 'Ridge' + ' BiVa_boot.pdf', multi_x=False)



def plot_MSE_resampling(n, MSE_boot, MSE_Kfold, regression_method):
    n_pol = MSE_Kfold.shape[0]
    poly_degs = np.arange(n_pol)


    plot_2D(poly_degs, [MSE_boot['test'],MSE_boot['train']], plot_count = 2, label = ['test','train'],
        title=regression_method + " MSE comparison " + str(n**2) + ' points',x_title="polynomial degree",y_title="MSE",filename= regression_method + ' MSE_comp.pdf', multi_x=False)

    #plot_2D(poly_degs, [MSE_boot['test'],MSE_boot['bias'],MSE_boot['variance']], plot_count = 3, label = ['MSE','bias','variance'],
    #    title=regression_method + " Bias-Variance " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= regression_method + ' BiVa_boot.pdf', multi_x=False)

    plot_2D(poly_degs, [MSE_boot['test'],MSE_boot['train'],MSE_Kfold], plot_count = 3, label = ['Test error','Training error', 'K-fold predictions'],
        title=regression_method + " Kfold prediction for test error " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= regression_method + ' Kfold_test.pdf', multi_x=False)


def make_predictions(models, x_test):
    """ Makes predictions on models without resampling

    :models: array_like of Model objects
    :x_test: array_like of tuples
    :returns: 2D array_like, list of arrays

    """
    n_pol = len(models)
    poly_degs = np.arange(n_pol)
    z_pred = np.empty((n_pol, len(x_test),1))
    betas = []

    for i, model in enumerate(models):
        z_pred[i] = model.predict(x_test)
        betas.append(model.beta)
    return z_pred, betas


def plot_MSE_R2(n,z_test, z_pred, regression_method):
    """ Plots MSE score as well as R2 score for models against their polynomial degree

    :z_test: array_like
    :z_pred: 2D array_like

    """

    n_pol = z_pred.shape[0]
    poly_degs = np.arange(n_pol)
    MSEs = np.empty(n_pol)
    R2s = np.empty(n_pol)
    for i in poly_degs:
        MSEs[i] = MSE(z_test,z_pred[i])
        R2s[i] = R2(z_test,z_pred[i])

    # Plots R2 score over polynomial degrees
    plot_2D(poly_degs, R2s, title=regression_method + " R$^2$ Score " + str(n**2) + ' points',x_title="polynomial degree",y_title="R2",filename= regression_method + ' R2.pdf')

    # Plots MSE score over polynomial degrees
    plot_2D(poly_degs,MSEs,title=regression_method + " Mean Squared Error" + str(n**2) + ' points', x_title="polynomial degree", y_title="MSE", filename=regression_method + ' MSE.pdf')


def plot_beta(n,betas,regression_method):
    """ Plots of a set of beta vectors against number of features in the corresponding model

    :betas: list of arrays

    """
    beta_ranges = [np.arange(len(beta)) for beta in betas]

    # Plots beta vectors for each polynomial degree, with number of features on x-axis
    plot_2D(beta_ranges, betas, plot_count = len(betas), title=regression_method + " beta " + str(n**2) + ' points',x_title="features",y_title="Beta",filename= regression_method + ' beta.pdf')


"""  Old code  (or not currently in use) """


"""
def plot_MSE_lasso(models, z_test, nlambdas):

    '''problematic shit!!'''

    z_train = model[-1].z_train
    poly_degs = np.arange(models[-1].polydeg + 1)
    for model in models:
        MSE_lasso_predicts = np.zeros(nlambdas)
        lambdas = np.logspace(-4, 4, nlambdas)
        X_train = model.X_dict["train"]
        X_test = model.X_dict["test"]

        for lamb in lambdas:
            reg_lasso = linear_model.Lasso(lamb)
            reg_lasso.fit(X_train, z_train)
            z_predict_lasso = reg_lasso.predict(whatever)
            print(reg_lasso.coef_)
            MSE_lasso_predicts[lamb] = MSE(z_test,z_predict_lasso)

    plot_2D([polydeg], [MSE_lasso_predicts], title='Lasso MSE', x_title='polynomial degrees',
            y_title='MSE', filename='lasso MSE')
"""
