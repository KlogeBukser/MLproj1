import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poly_funcs import get_2D_pols

from generate import *
from produce_results import *

np.random.seed(1)


# Max polynomial degree

n_boots = 20
n = 20
#polydeg = 8
#n_pol = polydeg + 1

# ols = False
# ridge = True
# lasso = False

x, z = generate_data_Franke(n, noise = 0.8)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Generates data, and splits it

def ols(polynomial_degree = 5):
    # Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)
    test_score, train_score, bias, var, Kfold_score = np.empty((5,n_pol))


    n_simple = np.min((6,n_pol))
    simple_degs = np.arange(n_simple)
    simple_MSE, simple_R2 = np.empty((2,n_simple))
    betas = []
    beta_ranges = []

    for deg in poly_degs:

        model = Model(deg)
        model.train(x_train,z_train)

        z_pred, z_fit = model.bootstrap(x_test,n_boots)

        test_score[deg] = MSE(z_test,z_pred)
        train_score[deg] = MSE(z_train,z_fit)
        bias[deg] = cal_bias(z_test, z_pred)
        var[deg] = cal_variance(z_pred)
        Kfold_score[deg] = model.cross_validate(6)


        if (deg < n_simple):
            betas.append(model.beta)
            beta_ranges.append(np.arange(model.feature_count))

            z_pred_simple = model.predict(x_test)

            simple_MSE[deg] = MSE(z_test,z_pred_simple)
            simple_R2[deg] = R2(z_test,z_pred_simple)



    # Plots R2 score over polynomial degrees
    plot_2D(simple_degs, simple_R2, title='ols' + " R$^2$ Score " + str(n**2) + ' points',x_title="polynomial degree",y_title="R2",filename= 'ols' + ' R2.pdf')

    # Plots MSE score over polynomial degrees
    plot_2D(simple_degs,simple_MSE,title='ols' + " Mean Squared Error" + str(n**2) + ' points', x_title="polynomial degree", y_title="MSE", filename='ols' + ' MSE.pdf')

    plot_2D(beta_ranges, betas, plot_count = len(betas), title='ols' + " beta " + str(n**2) + ' points',x_title="features",y_title="Beta",filename= 'ols' + ' beta.pdf')



    plot_2D(poly_degs, [test_score,bias,var], plot_count = 3, label = ['MSE','bias','variance'],
        title='ols' + " Bias-Variance " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= 'ols' + ' BiVa_boot.pdf', multi_x=False)

    plot_2D(poly_degs, [test_score,train_score], plot_count = 2, label = ['test','train'],
        title='ols' + " MSE comparison " + str(n**2) + ' points',x_title="polynomial degree",y_title="MSE",filename= 'ols' + ' MSE_comp.pdf', multi_x=False)

    plot_2D(poly_degs, [test_score,train_score,Kfold_score], plot_count = 3, label = ['Test error','Training error', 'K-fold predictions'],
        title='ols' + " Kfold prediction for test error " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= 'ols' + ' Kfold_test.pdf', multi_x=False)


def ridge():
    models = []


    for deg in range(n_pol):
        models.append(Ridge(deg))
        models[deg].train(x_train,z_train,'best')

    #plot_ridge(n,models,x_test,z_test,n_boots)
    pred,fit = make_predictions_boot(models,x_test,n_boots)
    MSE_test = find_test_MSE(z_test, pred)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)


    ols_z_pred, ols_z_fit = make_predictions_boot(models,x_test,n_boots)
    MSE_boot = find_MSE_boot(z_train, z_test, ols_z_pred, ols_z_fit)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)

    plot_MSE_resampling(n,MSE_boot,MSE_Kfold,'ridge')

    return
    n_lambdas = 20
    lambdas = np.logspace(-4,4,n_lambdas)
    test_score = np.empty((n_pol,n_lambdas))
    train_score = np.empty((n_pol,n_lambdas))

    for i,lamb in enumerate(lambdas):
        for model in models:
            model.set_lambda(lamb)

        pred, fit = make_predictions_boot(models,x_test,n_boots)
        scores = find_MSE_boot(z_train,z_test, pred, fit)
        test_score[:,i] = scores['test']
        train_score[:,i] = scores['train']

    plot_MSE_lambda(n,test_score,lambdas)


def lasso():

    nlambdas = 100
    lambdas = np.logspace(-4, 4, nlambdas)

    # for plotting
    mses = []
    mses_train = []
    labels = []

    # get design matrix

    for deg in range(polydeg + 1):
        functions = get_2D_pols(deg)
        func_count = len(functions)

        n = x_train.shape[0]
        X = np.ones((n, func_count))
        for i in range(n):
            for j in range(func_count):
                X[i,j] = functions[j](x_train[i])

        # will scale data with StandardScaler

        mse = np.zeros(nlambdas)
        mse_train = np.zeros(nlambdas)

        for i in range(nlambdas):
            lmb = lambdas[i]

            reg_lasso = make_pipeline(StandardScaler(with_mean=False), linear_model.Lasso(lmb))
            reg_lasso.fit(X,z_train)
            z_pred = reg_lasso.predict(X)

            mse[i] = MSE(z_test,z_pred)
            mse_train[i] = MSE(z_train,z_pred)

        mses.append(mse)
        mses_train.append(mse_train)
        labels.append('n = ' + str(deg))

    plot_lmb_MSE(np.log10(lambdas), mses, 'lasso', labels)
    plot_lmb_MSE(np.log10(lambdas), mses_train, 'lasso training', labels)


# calls
ols(8)
# ridge()
# lasso()

# part g
"""
terrain_datas = ['SRTM_data_Norway_1.tif', 'SRTM_data_Norway_2.tif']

for terrain_data in terrain_datas:

    xy,terrain = prep_terrain(terrain_data)
    x_train, x_test, z_train, z_test = train_test_split(xy,terrain)
    # TODO: 3 regression, cross-validation

    # SUPER SLOW!!!
    # ols()
"""
