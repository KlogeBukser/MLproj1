

from plotting import *
from calculate import *
from poly_funcs import *
from transform import *
from model import *

import numpy as np
from sklearn.model_selection import train_test_split


def plot_MSE_comparison(x, z, polydeg = 5, n_boots = 100):
    # Makes models for every polynomial degree up to the input
    # Uses bootstrap method for resampling
    # Plots MSE on the Test data, entire Training data
    # Also plots MSE for the bootstrap subsets

    # Empty containers for MSE values
    MSE_fit = np.zeros(polydeg + 1)
    MSE_pred = np.zeros(polydeg + 1)
    MSE_boot = np.zeros(polydeg + 1)

    # Makes a model that finds, and holds the polynomial features
    # The features are used to construct the design matrix
    model = Model(polydeg)
    X = model.design(x)

    # Split into training and testing data
    X_train, X_test, z_train, z_test = train_test_split(X,z)

    # Array of polynomial degrees for plotting
    poly_degrees = np.arange(0,polydeg + 1)

    for deg in poly_degrees[::-1]:
        # Loops backwards over polynomial degrees
        # Reduces model complexity with one polynomial degree per iteration
        # Loop is stopped from if-condition when it's not able to reduce complexity further

        for i in range(n_boots):
            # Collects subsets (with overlap) from training data
            # Fits model to the 'boot data'
            # Predicts z for the test set, training set, and boot set
            # Note that the test, and training sets are the same for every (inner) loop
            # but the boot set is different each time
            X_boot, z_boot = bootstrap(X_train,z_train)
            model.fit(X_boot, z_boot)
            z_fit = model.predict(X_train)
            z_pred = model.predict(X_test)
            z_fit_boot = model.predict(X_boot)

            # Calculates MSE on all 3 of these data sets from the predictions
            MSE_fit[deg] += MSE(z_train, z_fit)
            MSE_pred[deg] += MSE(z_test, z_pred)
            MSE_boot[deg] += MSE(z_boot,z_fit_boot)

        # Reduces the complexity of the model
        # Stops the loop if the model complexity cannot be reduced further
        if not(model.reduce_complexity()):
            break


    # Scales the MSE values to get the average over bootstraps
    MSE_fit /= n_boots
    MSE_pred /= n_boots
    MSE_boot /= n_boots

    # Plots and saves plot of MSE comparisons
    multi_yplot(range(polydeg + 1), (MSE_pred,MSE_fit,MSE_boot),("Testing","Training","Bootstrap data"), "Mean Squared Error", "polynomial degree", "Score")
    plt.savefig("plots/MSE_comp.pdf")
    plt.close()



def plot_scores_beta(x, z, polydeg = 5, axis_features = True):

    # Produces plot(s) for measuring quality of 2D polynomial model
    # Plots Mean squared error, R2-score, and beta values
    # The Beta plot uses features on the x-axis, but numbers can be used instead if (features_beta = False)

    MSE_vals = np.zeros(polydeg + 1)
    R2_vals = np.zeros(polydeg + 1)

    # Makes a model that finds, and holds the polynomial features
    # The features are used to construct the design matrix
    model = Model(polydeg)
    X = model.design(x)

    # Split into training and testing data
    X_train, X_test, z_train, z_test = train_test_split(X,z)

    # Array of polynomial degrees for plotting
    poly_degs = np.arange(0,polydeg + 1,1)

    for deg in poly_degs[::-1]:
        # Loops backwards over polynomial degrees
        # Fits model to training data
        # Makes prediction for z, and compares to test data with MSE and R squared
        model.fit(X_train, z_train)
        z_pred = model.predict(X_test)
        MSE_vals[deg] = MSE(z_test, z_pred)
        R2_vals[deg] = R2(z_test, z_pred)

        # Collects beta from the model
        # Makes overlapping plots of beta, one for each polynomial degree
        beta = model.get_beta()
        if (axis_features):
            # Collects the polynomial as a string to use as the x-axis
            # While neat, this might be too clunky
            x_ax = get_2D_string(deg)
        else:
            # Uses integers on the x-axis
            # This is more readable, and should be used on large polynomials
            x_ax = np.arange(1,len(beta) + 1)
        plt.plot(x_ax,beta, label = "Degree %d" % deg)

        # Reduces the complexity of the model
        # Stops the loop if the model complexity cannot be reduced further
        if not(model.reduce_complexity()):
            break


    # Saves the overlapping Beta plots to file
    title_axlabs("Beta for different amounts of features","Features","Beta")
    plt.legend()
    plt.savefig("plots/beta.pdf")
    plt.close()

    # Plots R2 score over polynomial degrees
    simple_plot(poly_degs,R2_vals, "R$^2$ Score", "polynomial degree", "R2")
    plt.savefig("plots/R2.pdf")
    plt.close()

    # Plots MSE score over polynomial degrees
    simple_plot(poly_degs,MSE_vals,"Mean Squared Error", "polynomial degree", "MSE")
    plt.savefig("plots/MSE.pdf")
    plt.close()
