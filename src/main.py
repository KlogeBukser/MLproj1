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
polydeg = 8
n_boots = 100
n = 20

# ols = False
# ridge = True
# lasso = False

x, z = generate_data_Franke(n, noise = 0.4)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Generates data, and splits it

def ols():
    # Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions
    models = []
    for deg in range(polydeg + 1):
        models.append(Model(deg))
        models[deg].train(x_train, z_train)

    # Finds prediction values without resampling
    z_pred, betas = make_predictions(models[:6], x_test)

    # Plots desired values for the model without resampling
    plot_MSE_R2(n,z_test, z_pred, regression_method = 'ols')
    plot_beta(n,betas, regression_method = 'ols')


    # Finds prediction values with the bootstrap method
    z_pred_b, z_fit_b = make_predictions_boot(models,x_test,n_boots)

    # Plots desired values for the (bootstrap) resampled predictions
    MSE_boot = find_MSE_boot(z_train, z_test, z_pred_b, z_fit_b)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)

    plot_MSE_resampling(n,MSE_boot,MSE_Kfold,'ols')

def ridge():
    models = []
    for deg in range(polydeg + 1):
        models.append(Ridge(deg))
        models[deg].train(x_train,z_train,'best')

    z_pred_b, z_fit_b = make_predictions_boot(models,x_test,n_boots)
    MSE_boot = find_MSE_boot(z_train, z_test, z_pred_b, z_fit_b)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)

    plot_MSE_resampling(n,MSE_boot,MSE_Kfold,'ridge')

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
# ols()
# ridge()
lasso()

# part g
terrain_datas = ['SRTM_data_Norway_1.tif', 'SRTM_data_Norway_2.tif']

for terrain_data in terrain_datas:

    xy,terrain = prep_terrain(terrain_data)
    x_train, x_test, z_train, z_test = train_test_split(xy,terrain)
    # TODO: 3 regression, cross-validation

    # SUPER SLOW!!!
    # ols()


