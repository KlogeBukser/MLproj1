
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

def find_MSE_Kfold(models, k_vals):

    n_pol = models[-1].polydeg + 1
    if models[0].algorithms.regression_method == 'ridge':
        print("stop, this isn't ready yet")
        return

    MSEs = np.empty((len(k_vals), n_pol))
    for i, model in enumerate(models):
        for j, k in enumerate(k_vals):
            MSEs[j,i] = model.cross_validate(k)



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


def find_MSE_boot(z_train,z_test,z_pred,z_fit, include_bias_var = True):
    """ Plots MSE score for models against their polynomial degree on test set and training set.
    For bootstrap predictions


    :z_test: array_like
    :z_pred: 2D array_like

    """

    n_pol = z_pred.shape[0]
    MSE_dict = make_container(['test','train','bias','variance'],n_pol)
    poly_degs = np.arange(n_pol)
    for i in poly_degs:
        MSE_dict['test'][i] = MSE(z_test, z_pred[i])
        MSE_dict['train'][i] = MSE(z_train,z_fit[i])
        MSE_dict['bias'][i] = cal_bias(z_test,z_pred[i])
        MSE_dict['variance'][i] = cal_variance(z_pred[i])

    return MSE_dict

def find_MSE(z_train,z_test,z_pred,z_fit):
    n_pol = z_pred.shape[0]
    MSE_dict = make_container(['test','train'],n_pol)
    poly_degs = np.arange(n_pol)



def plot_MSE_resampling(n, MSE_boot, MSE_Kfold, regression_method):
    n_pol = MSE_Kfold.shape[1]
    poly_degs = np.arange(n_pol)
    K_comp = [MSE_boot['test'],MSE_boot['train']] + [MSE_Kfold[i] for i in range(MSE_Kfold.shape[0])]

    plot_2D(poly_degs, [MSE_boot['test'],MSE_boot['train']], plot_count = 2, label = ['test','train'],
        title=regression_method + " MSE comparison " + str(n**2) + ' points',x_title="polynomial degree",y_title="MSE",filename= regression_method + ' MSE_comp.pdf', multi_x=False)

    plot_2D(poly_degs, [MSE_boot['test'],MSE_boot['bias'],MSE_boot['variance']], plot_count = 3, label = ['MSE','bias','variance'],
        title=regression_method + " Bias-Variance " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= regression_method + ' BiVa_boot.pdf', multi_x=False)

    plot_2D(poly_degs, K_comp, plot_count = len(K_comp), label = ['Test error','Training error','K5','K6','K7','K8','K9','K10'],
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
def produce_error(data, model):
    '''produce MSE, R2 erros for any set of data for any given regression model.
    input: data: array like,
    return tuple of 2 arrays in which the first is a mse and the seocnd is r2'''

    mse_err = MSE(data, model)
    r2_err = R2(data, model)
    var = cal_variance(data, model)
    bias = cal_bias(data, model)

    return mse_err, r2_err
"""

"""
def plot_MSE_comparison(models, z_test, n, regression_method = 'ols'):
    ''' Makes models for every polynomial degree up to the input
    Uses bootstrap method for resampling
    Plots MSE on the Test data, entire Training data, and bootstrap samples '''

    # model, poly_degs, MSE_dict, x_train, x_test, z_train, z_test = train_model(polydeg, x, z, ['pred','fit',resample_method], regression_method)
    n_pol = models[-1].polydeg + 1
    poly_degs = np.arange(n_pol)
    z_train = models[-1].z_train

    err_dict = make_container(['Test','Train','k1','k2','k3','k4','k5'],n_pol)


    for model in models:
        z_pred = model.predict("test")
        z_fit = model.predict("train")
        deg = model.polydeg
        err_dict['Test'][deg] = MSE(z_test, z_pred)
        err_dict['Train'][deg] = MSE(z_train, z_fit)
        kfold_scores = np.mean(model.cross_validate(5,n_lambs = 5), keepdims = True, axis = 1)
        print(kfold_scores)
        for i in range(5):
            err_dict['k%d' % (i+1)][deg] = kfold_scores[i]

    # Plots and saves plot of MSE comparisons
    plot_2D(poly_degs, list(err_dict.values()), plot_count = 7, label = list(err_dict.keys()),
        title=regression_method + " MSE comparison " + str(n**2) + ' points' ,x_title="polynomial degree",y_title="MSE",
        filename= regression_method + ' MSE_comp.pdf', multi_x=False)
"""

"""
def plot_MSEs(models, z_test, nlambdas=100,regression_method='ols', resample_method='boot', n=20):

    print("Regression method : ", regression_method)
    assert regression_method in ALLOWED_METHODS, ERR_INVALID_METHOD

    if regression_method == 'lasso':
        plot_MSE_lasso(models, z_test, nlambdas,n) # does nothing rn

    else:
        plot_MSE_comparison(models, z_test, n, regression_method=regression_method)
"""

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
