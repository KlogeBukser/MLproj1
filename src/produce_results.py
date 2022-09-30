
from plotting import *
from calculate import *
from transform import *
from model import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

ALLOWED_METHODS = ['ols','ridge','lasso']

ERR_INVALID_METHOD = 'Invalid regression_method, allowed methods are {0}, {1}, {2}'
ERR_INVALID_METHOD.format(*ALLOWED_METHODS)

def produce_error(data, model):
    '''produce MSE, R2 erros for any set of data for any given regression model.
    input: data: array like,
    return tuple of 2 arrays in which the first is a mse and the seocnd is r2'''

    mse_err = MSE(data, model)
    r2_err = R2(data, model)
    var = cal_variance(data, model)
    bias = cal_bias(data, model)

    return mse_err, r2_err

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

def make_predictions_boot(models,prediction_names,n_boots = 100):
    """ Uses bootstrap method to find predictions (z_pred,z_fit, etc)
    :models:
    :n_boots:
    :prediction_names:
    :regression_method:
    :return:
    """
    predictions = make_container(prediction_names)
    for model in models:
        model.start_boot(n_boots, predict_boot = False)
        for name in prediction_names:
            predictions[name].append(model.boot_predict(name))
        model.end_boot()
    return predictions


def plot_MSE_comparison2(z_data, z_predicts, regression_method = 'ols'):
    ''' Under development
    This method is functional, but messy at the moment.
    It does the same thing as plot plot_MSE_comparison, but does not use the models at all.
    This lets us skip making, and training new models for every plot we need'''

    n_pol = len(z_predicts['test'])
    poly_degs = np.arange(n_pol)
    datasets = z_data.keys()
    MSE_dict = make_container(datasets,n_pol)
    n_boots = z_predicts["test"][0].shape[1]
    for deg in poly_degs:
        for name in datasets:
            MSE_dict[name][deg] = np.mean([MSE(z_data[name], z_predicts[name][deg][:,i]) for i in range(n_boots)])

    # Plots and saves plot of MSE comparisons
    plot_2D(poly_degs, list(MSE_dict.values()), plot_count = 2, label = list(MSE_dict.keys()),
        title=regression_method + " MSE comparison ",x_title="polynomial degree",y_title="MSE",filename= regression_method + ' MSE_comp.pdf', multi_x=False)


def plot_MSE_comparison(models, z_test, n_boots = 100, regression_method = 'ols', resample_method='boot'):
    ''' Makes models for every polynomial degree up to the input
    Uses bootstrap method for resampling
    Plots MSE on the Test data, entire Training data, and bootstrap samples '''

    # model, poly_degs, MSE_dict, x_train, x_test, z_train, z_test = train_model(polydeg, x, z, ['pred','fit',resample_method], regression_method)
    n_pol = models[-1].polydeg + 1
    poly_degs = np.arange(n_pol)
    z_train = models[-1].z_train

    MSE_dict = make_container(['Test','Train',resample_method],n_pol)

    if resample_method == 'boot':
        for model in models:
            z_boot,z_boot_fit = model.start_boot(n_boots, predict_boot = True)
            z_pred = model.boot_predict("test")
            z_fit = model.boot_predict("train")
            model.end_boot()

            deg = model.polydeg
            MSE_dict['Test'][deg] = np.mean([MSE(z_test, z_pred[:,i]) for i in range(n_boots)])
            MSE_dict['Train'][deg] = np.mean([MSE(z_train, z_fit[:,i]) for i in range(n_boots)])
            MSE_dict[resample_method][deg] = np.mean([MSE(z_boot[:,i],z_boot_fit[:,i]) for i in range(n_boots)])


    # Plots and saves plot of MSE comparisons
    plot_2D(poly_degs, list(MSE_dict.values()), plot_count = 3, label = list(MSE_dict.keys()),
        title=regression_method + " MSE comparison ",x_title="polynomial degree",y_title="MSE",filename= regression_method + ' MSE_comp.pdf', multi_x=False)



def plot_scores_beta(models, z_test, regression_method='ols'):

    '''Produces plot(s) for measuring quality of 2D polynomial model
    Plots Mean squared error, R2-score, and beta values
    The Beta plot uses features on the x-axis, but numbers can be used instead if (features_beta = False) '''

    n_pol = models[-1].polydeg + 1
    poly_degs = np.arange(n_pol)
    ['MSE', 'R2']
    score_dict = make_container(['MSE', 'R2'], n_pol)
    betas = []
    beta_ranges = []

    for model in models:
        deg = model.polydeg

        z_pred = model.predict("test")
        score_dict['MSE'][deg] = MSE(z_test, z_pred)
        score_dict['R2'][deg] = R2(z_test, z_pred)

        betas.append(model.beta)
        beta_ranges.append(range(model.feature_count))

    # Plots beta vectors for each polynomial degree, with number of features on x-axis
    plot_2D(beta_ranges, betas, plot_count = len(betas), title=regression_method + " beta ",x_title="features",y_title="Beta",filename= regression_method + ' beta.pdf')

    # Plots R2 score over polynomial degrees
    plot_2D(poly_degs, score_dict['R2'], title=regression_method + " R$^2$ Score ",x_title="polynomial degree",y_title="R2",filename= regression_method + ' R2.pdf')

    # Plots MSE score over polynomial degrees
    plot_2D(poly_degs,score_dict['MSE'],title=regression_method + " Mean Squared Error", x_title="polynomial degree", y_title="MSE", filename=regression_method + ' MSE.pdf')

def plot_bias_var(is_resemble=False, resample_method=None):

    pass

def plot_MSEs(models, z_test, n_boots=100, nlambdas=100, regression_method='ols', resample_method='boot'):

    print("Regression method : ", regression_method)
    assert regression_method in ALLOWED_METHODS, ERR_INVALID_METHOD

    if regression_method == 'lasso':
        plot_MSE_lasso(models, z_test, nlambdas) # does nothing rn

    else:
        plot_MSE_comparison(models, z_test, regression_method=regression_method)


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
