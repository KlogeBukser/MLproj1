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

resampling_method = 'boot'
# resampling_method = 'cross'

# Max polynomial degree
polydeg = 5

# Generates data, and splits it
x, z = generate_data_Franke(20,noise = 0.5)
x_train, x_test, z_train, z_test = train_test_split(x,z)

# Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions
models = []
for deg in range(polydeg + 1):
    models.append(Model(deg, x_train, z_train, regression_method = regression_method))
    models[deg].add_x(x_test,"test")
    
# choose desired plots
plot_MSEs(models, z_test, regression_method=regression_method) # plots MSE as a function of polynomial degrees for both no resampling and bootstrap


#this doesn't work for lasso since there's no analytical expression for beta_lasso
plot_scores_beta(models,z_test,regression_method=regression_method) # plots beta, R2 scores and MSE as a function of features
