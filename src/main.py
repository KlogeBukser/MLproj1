import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from generate import generate_data_Franke
from produce_results import *

np.random.seed(1) # To have better control of what the plots are supposed to look like


# choose regression methods

regression_method = 'ols'
# regression_method = 'ridge'
# regression_method = 'lasso'


# choose resampling methods

# resampling_method = 'boot'
# resampling_method = 'cross'

# Max polynomial degree
polydeg = 5
n_boots = 5

# Generates data, and splits it
x, z = generate_data_Franke(20,noise = 0.5)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions
boot_models = []
none_models = []
for deg in range(polydeg + 1):
    boot_models.append(Model(deg, x_train, z_train, resampling_method='boot', regression_method = regression_method, n_res = n_boots))
    boot_models[deg].add_x(x_test,"test")

    none_models.append(Model(deg, x_train, z_train, resampling_method='none', regression_method = regression_method))
    none_models[deg].add_x(x_test,"test")


"""This block of code is under development, but functional
It finds the z predictions first, and use those to find MSE (and eventually other stuff)
Atm it does the same as just calling plot_MSEs

prediction_names = ["train","test"]
predictions = make_predictions_boot(models,prediction_names,n_boots = n_boots)
data = make_container(prediction_names)
data["train"] = z_train
data["test"] = z_test
plot_MSE_comparison2(data, predictions, regression_method = regression_method)
"""

# choose desired plots
plot_MSEs(boot_models, z_test, regression_method=regression_method) # plots MSE as a function of polynomial degrees for both no resampling and bootstrap


#this doesn't work for lasso since there's no analytical expression for beta_lasso
plot_scores_beta(none_models,z_test,regression_method=regression_method) # plots beta, R2 scores and MSE as a function of features
