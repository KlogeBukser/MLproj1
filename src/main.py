import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from produce_results import *

np.random.seed(1)

x, z = generate_data_Franke(20,noise = 0.5)

# for ols 
plot_MSE_comparison(x,z,8)
plot_scores_beta(x,z,5)

# for ridge
# plot_MSE_comparison(x,z,8,regression_method='ridge')
# plot_scores_beta(x,z,5,regression_method='ridge')

# for lasso

