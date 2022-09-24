from plotting import *
from calculate import *
from poly_funcs import *
from transform import *

import numpy as np

def prod_score_plots(feature_values, label_values, max_poly = 5, include_MSE = True, include_R2 = False):
    # (x,y) = features
    # z = label
    # Produces plot(s) for measuring quality of model
    # Plots Mean squared error by default, but R2 can be plotted as well/instead (in separate plot)

    poly_degs = np.arange(1,max_poly + 1)

    MSE_vals = np.zeros(max_poly)
    R2_vals = np.zeros(max_poly)

    for i,deg in enumerate(poly_degs):

        funcs = get_2D_pols(deg)
        design = calc_design(feature_values,funcs)
        X_train, X_test, z_train, z_test = split_scale(design,label_values)
        z_tilde = get_model(X_train,z_train)
        R2_vals[i] = R2(z_train,z_tilde)
        MSE_vals[i] = MSE(z_train,z_tilde)

    if (include_R2):
        simple_plot(poly_degs,R2_vals, "R$^2$")

    if (include_MSE):
        simple_plot(poly_degs,MSE_vals,"Mean Squared Error")
