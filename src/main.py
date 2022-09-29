import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from generate import *
from produce_results import *

np.random.seed(1) # ??


# choose regression methods

regression_method = 'ols'
# regression_method = 'ridge'
# regression_method = 'lasso'


# choose resampling methods

resampling_method = 'boot'
# resampling_method = 'cross'

# Max polynomial degree
polydeg = 5

# Generates data, and splits it
x, z = generate_data_Franke(20,noise = 0.5)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Makes model, and feeds it the testing data (x_test) for predictions
model = Model(polydeg, x_train, z_train, regression_method=regression_method)
model.add_x(x_test,"test")

# choose desired plots
plot_MSEs(model, z_test, regression_method=regression_method) # plots MSE as a function of polynomial degrees for both no resampling and bootstrap


""" Extra model until restructure """
model2 = Model(polydeg, x_train, z_train, regression_method=regression_method)
model2.add_x(x_test,"test")

#this dones't work for lasso since there's no analytical expression for beta_lasso
plot_scores_beta(model2,z_test,regression_method=regression_method) # plots beta, R2 scores and MSE as a function of features
