from plotting import *
from calculate import *
from poly_funcs import *
from transform import *

import numpy as np

def prod_score_plots(feature_values, label_values, max_poly = 5, include_MSE = True, include_R2 = True, include_beta = True):
    # (x,y) = features
    # z = label
    # Produces plot(s) for measuring quality of 2D polynomial model
    # Plots Mean squared error, R2-score, and beta values
    # Each plot can be disabled by optional boolean parameters

    poly_degs = np.arange(1,max_poly + 1)

    MSE_vals = np.zeros(max_poly)
    R2_vals = np.zeros(max_poly)

    for i,deg in enumerate(poly_degs):

        funcs = get_2D_pols(deg)
        design = calc_design(feature_values,funcs)
        X_train, X_test, z_train, z_test = split_scale(design,label_values)
        beta = find_beta(X_train,z_train)
        z_tilde = get_model(X_train,beta)
        R2_vals[i] = R2(z_train,z_tilde)
        MSE_vals[i] = MSE(z_train,z_tilde)
        if (include_beta):
            points = np.arange(1,len(beta) + 1, 1)
            plt.plot(points,beta, label = "Polynomials up to degree %d" % deg)

    if (include_beta):
        title_axlabs("beta values when adding more features","Features","beta")
        plt.legend()
        plt.show()

    if (include_R2):
        simple_plot(poly_degs,R2_vals, "R$^2$", "polynomial degree", "Score")

    if (include_MSE):
        simple_plot(poly_degs,MSE_vals,"Mean Squared Error", "polynomial degree", "Score")
