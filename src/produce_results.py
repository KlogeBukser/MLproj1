
from plotting import *
from calculate import *
from poly_funcs import *
from transform import *
from model import *

import numpy as np
from sklearn.model_selection import train_test_split


def produce_error(data, model):
    '''produce MSE, R2 erros for any set of data for any given regression model.
    input: data: array like, 
    return tuple of 2 arrays in which the first is a mse and the seocnd is r2'''

    mse_err = MSE(data, model)
    r2_err = R2(data, model)

    return mse_err, r2_err

def plot_MSE_comparison(x, z, polydeg = 5, n_boots = 100, regression_method='ols'):
    # Makes models for every polynomial degree up to the input
    # Uses bootstrap method for resampling
    # Plots MSE on the Test data, entire Training data, and bootstrap samples

    # Number of relevant polynomial degrees (it includes 0)
    n_pol = polydeg + 1

    # Split into training and testing data
    x_train, x_test, z_train, z_test = train_test_split(x,z)

    # Makes a model that
    # -Holds polynomial features
    # -Constructs and holds design matrix, and training z
    # -Fits the model by finding and saving beta
    model = Model(polydeg,x_train,z_train)

    # Adds the testing data x_test to the dictionary with key "test"
    model.add_x(x_test, "test")

    # Array of polynomial degrees
    poly_degs = np.arange(n_pol)

    # Empty containers for MSE values
    MSE_fit = np.zeros(n_pol)
    MSE_pred = np.zeros(n_pol)
    MSE_boot = np.zeros(n_pol)

    for deg in poly_degs[::-1]:
        # Loops backwards over polynomial degrees
        # Reduces model complexity with one polynomial degree per iteration

        # try absorb the next 7 lines into model
        z_boot,z_boot_fit = model.start_boot(n_boots, True, regression_method)
        # z_boot,z_boot_fit = model.start_boot(n_boots,predict_boot = True, regression_method = 'ridge')
        z_pred = model.boot_predict("test")
        z_fit = model.boot_predict("train")
        model.end_boot()

        MSE_pred[deg] = np.mean([MSE(z_test, z_pred[:,i]) for i in range(n_boots)])
        MSE_fit[deg] = np.mean([MSE(z_train, z_fit[:,i]) for i in range(n_boots)])
        MSE_boot[deg] = np.mean([MSE(z_boot[:,i],z_boot_fit[:,i]) for i in range(n_boots)])

        # Reduces the complexity of the model
        model.reduce_complexity()


    # Plots and saves plot of MSE comparisons
    multi_yplot(poly_degs, (MSE_pred,MSE_fit,MSE_boot),("Testing","Training","Bootstrap data"), regression_method + " Mean Squared Error", "polynomial degree", "Score")
    plt.savefig("plots/" + regression_method + " MSE_comp.pdf")
    plt.show()


def plot_scores_beta(x, z, polydeg = 5, axis_features = True, regression_method='ols'):

    '''Produces plot(s) for measuring quality of 2D polynomial model
    Plots Mean squared error, R2-score, and beta values
    The Beta plot uses features on the x-axis, but numbers can be used instead if (features_beta = False) '''

    # Number of relevant polynomial degrees (it includes 0)
    n_pol = polydeg + 1

    MSE_vals = np.zeros(n_pol)
    R2_vals = np.zeros(n_pol)

    # Makes a model that finds, and holds the polynomial features
    # The features are used to construct the design matrix


    # Split into training and testing data
    x_train, x_test, z_train, z_test = train_test_split(x,z)
    model = Model(polydeg,x_train,z_train)
    model.add_x(x_test,"test")

    # Array of polynomial degrees for plotting
    poly_degs = np.arange(n_pol)

    for deg in poly_degs[::-1]:
        # Loops backwards over polynomial degrees
        # Fits model to training data
        # Makes prediction for z, and compares to test data with MSE and R squared
        #model.fit(X_train, z_train)
        z_pred = model.predict("test")
        MSE_vals[deg] = MSE(z_test, z_pred)
        R2_vals[deg] = R2(z_test, z_pred)

        # Collects beta from the model
        # Makes overlapping plots of beta, one for each polynomial degree
        beta = model.choose_beta(model.X_dict["train"], model.z, regression_method)

        if (axis_features):
            # Collects the polynomial as a string to use as the x-axis
            # While neat, this might be too clunky
            x_ax = get_2D_string(deg)
        else:
            # Uses integers on the x-axis
            # This is more readable, and should be used on large polynomials
            x_ax = np.arange(len(beta))
        plt.plot(x_ax, beta, label = "Degree %d" % deg)

        # Reduces the complexity of the model
        # Stops the loop if the model complexity cannot be reduced further
        model.reduce_complexity()


    # Saves the overlapping Beta plots to file

    set_paras(title="Beta " + regression_method + " for different amounts of features",x_title='Features',y_title='Beta',
        filename = regression_method + ' beta.pdf', file_dir='plots')

    # Plots R2 score over polynomial degrees
    plot_2D([poly_degs], [R2_vals], title=regression_method + " R$^2$ Score ",x_title="polynomial degree",y_title="R2",filename= regression_method + ' R2.pdf')

    # Plots MSE score over polynomial degrees
    plot_2D([poly_degs],[MSE_vals],title=regression_method + " Mean Squared Error", x_title="polynomial degree", y_title="MSE", filename=regression_method + ' MSE.pdf')
