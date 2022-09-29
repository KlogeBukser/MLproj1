import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from produce_results import *

np.random.seed(1) # ??

x, z = generate_data_Franke(20,noise = 0.5)

# choose regression methods

# regression_method = 'ols'
# regression_method = 'ridge'
regression_method = 'lasso'


# choose resampling methods

resampling_method = 'boot'
# resampling_method = 'cross'


# choose desired plots
plot_MSEs(x, z, 5, regression_method=regression_method) # plots MSE as a function of polynomial degrees for both no resampling and bootstrap 

#this dones't work for lasso since there's not analytical expression for beta_lasso
plot_scores_beta(x,z,5,regression_method=regression_method) # plots beta, R2 scores and MSE as a function of features 


