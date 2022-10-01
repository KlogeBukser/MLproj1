import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from generate import generate_data_Franke
from produce_results import *

np.random.seed(1)


# choose regression methods

regression_method = 'ols'
# regression_method = 'ridge'
# regression_method = 'lasso'


# choose resampling methods

# resampling_method = 'boot'
# resampling_method = 'cross'

# Max polynomial degree
polydeg = 5
n_boots = 100

# Generates data, and splits it
for n in (20,30,40):
    x, z = generate_data_Franke(n,noise = 0.5)
    x_train, x_test, z_train, z_test = train_test_split(x,z)


    # Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions
    models = []
    for deg in range(polydeg + 1):
        models.append(Model(deg, regression_method = regression_method))
        models[deg].fit(x_train,z_train)


    # Finds prediction values without resampling
    z_pred, betas = make_predictions(models, x_test)

    # Plots desired values for the model without resampling
    plot_MSE_R2(z_test, z_pred)
    plot_beta(betas)

    # Finds prediction values with the bootstrap method
    z_pred_b, z_fit_b = make_predictions_boot(models,x_test,n_boots)

    # Plots desired values for the (bootstrap) resampled predictions
    plot_boot(z_train, z_test, z_pred_b, z_fit_b)


    break
    #plot_MSEs(boot_models, z_test, regression_method=regression_method, n=n) # plots MSE as a function of polynomial degrees for both no resampling and bootstrap
