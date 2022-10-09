import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from generate import *
from produce_results import *

np.random.seed(1)


# Max polynomial degree
polydeg = 8
n_boots = 100
n = 20

ols = False
ridge = True
lasso = False

x, z = generate_data_Franke(n, noise = 0.4)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Generates data, and splits it
if (ols):
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

if (ridge):
    models = []
    for deg in range(polydeg + 1):
        models.append(Ridge(deg))
        models[deg].train(x_train,z_train,'best')

    z_pred_b, z_fit_b = make_predictions_boot(models,x_test,n_boots)
    MSE_boot = find_MSE_boot(z_train, z_test, z_pred_b, z_fit_b)
    MSE_Kfold = find_MSE_Kfold(models, folds = 6)

    plot_MSE_resampling(n,MSE_boot,MSE_Kfold,'ridge')


# part g
terrain_datas = ['SRTM_data_Norway_1.tif', 'SRTM_data_Norway_2.tif']

for terrain_data in terrain_datas:
    x,terrain = prep_terrain(terrain_data)
    continue

# test part g
# x,terrain = prep_terrain('SRTM_data_Norway_1.tif')
# print(x.shape)
# print(terrain.shape)


