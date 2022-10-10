import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poly_funcs import get_2D_pols
from model import *
from calculate import *

from generate import *
from produce_results import *

np.random.seed(1)


n_boots = 20
n = 20


x, z = generate_data_Franke(n, noise = 0.8)
x_train, x_test, z_train, z_test = train_test_split(x,z)



def ols(polynomial_degree = 5):
    # Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)
    test_score, train_score, bias, var, Kfold_score = np.empty((5,n_pol))

    n_simple = np.min((6,n_pol))
    simple_degs = np.arange(n_simple)
    simple_MSE, simple_R2 = np.empty((2,n_simple))
    betas = []

    for deg in poly_degs:

        model = Model(deg)
        model.train(x_train,z_train)

        z_pred, z_fit = model.bootstrap(x_test,n_boots)

        test_score[deg] = MSE(z_test,z_pred)
        train_score[deg] = MSE(z_train,z_fit)
        bias[deg] = cal_bias(z_test, z_pred)
        var[deg] = cal_variance(z_pred)
        Kfold_score[deg] = model.cross_validate(6)


        if (deg < n_simple):

            betas.append(model.beta)
            z_pred_simple = model.predict(x_test)
            simple_MSE[deg] = MSE(z_test,z_pred_simple)
            simple_R2[deg] = R2(z_test,z_pred_simple)

    # Plots Beta
    plot_beta(n,betas)
    # Plots MSE + R2
    plot_simple_scores(n,simple_MSE,simple_R2)
    # Plots KFold, bias/variance, test/train comparison
    plot_boot_scores(n,poly_degs,test_score,train_score,bias,var,Kfold_score)


def ridge(polynomial_degree = 5):

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)
    ridge_score, ols_score, k_ols_score = np.empty((3,n_pol))


    n_lambdas = 20
    lambdas = np.logspace(-4,4,n_lambdas)
    polydeg_lam = [0,1,3,6,8]
    test_score_lam = np.empty((len(polydeg_lam),n_lambdas))
    deg_lamb = 0


    for deg in poly_degs:

        model = Ridge(deg)
        model.train(x_train,z_train,'best')

        z_pred, z_fit = model.bootstrap(x_test,n_boots)
        ridge_score[deg] = MSE(z_test,z_pred)
        #k_ridge_score[deg] = model.cross_validate(6)

        model.set_lambda(0)
        z_pred, z_fit = model.bootstrap(x_test,n_boots)
        ols_score[deg] = MSE(z_test,z_pred)
        k_ols_score[deg] = model.cross_validate(6)

        if (deg in polydeg_lam):

            for i,lamb in enumerate(lambdas):
                model.set_lambda(lamb)
                pred, fit = model.bootstrap(x_test,n_boots)
                test_score_lam[deg_lamb,i] = MSE(z_test,pred)
            deg_lamb += 1


    plot_2D(np.log10(lambdas), test_score_lam, plot_count = 5, label = ['p = ' + str(i) for i in polydeg_lam],
        title='Ridge' + " Test-MSE " + str(n**2) + ' points',x_title="$\lambda$",y_title="Error",filename= 'Ridge' + ' BiVa_boot.pdf', multi_x=False)


    plot_2D(poly_degs, [ridge_score,ols_score], plot_count = 2, label = ['Ridge','OLS'],
        title='Ridge' + " OLS comparison " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= 'Ridge' + ' BiVa_boot.pdf', multi_x=False)


    plot_2D(poly_degs, [ridge_score,ols_score,k_ols_score], plot_count = 3, label = ['Ridge','OlS', 'K-OLS'],
        title='Ridge' + " Kfold prediction for test error " + str(n**2) + ' points',x_title="polynomial degree",y_title="Error",filename= 'ridge' + ' Kfold_test.pdf', multi_x=False)



def lasso(polynomial_degree = 5):

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)

    nlambdas = 100
    lambdas = np.logspace(-4, 4, nlambdas)

    # for plotting
    mses = []
    mses_train = []
    labels = []

    # get design matrix

    for deg in poly_degs:
        functions = get_2D_pols(deg)
        func_count = len(functions)

        n = x_train.shape[0]
        X = np.ones((n, func_count))
        for i in range(n):
            for j in range(func_count):
                X[i,j] = functions[j](x_train[i])

        n_test = x_test.shape[0]
        X_test = np.ones((n_test, func_count))
        for i in range(n_test):
            for j in range(func_count):
                X[i,j] = functions[j](x_test[i])

        # will scale data with StandardScaler

        mse = np.zeros(nlambdas)
        mse_train = np.zeros(nlambdas)

        for i in range(nlambdas):
            lmb = lambdas[i]

            reg_lasso = make_pipeline(StandardScaler(with_mean=False), linear_model.Lasso(lmb))
            reg_lasso.fit(X,z_train)
            z_pred = reg_lasso.predict(X)
            z_pred_test = reg_lasso.predict(X_test)

            mse[i] = MSE(z_test,z_pred_test)
            mse_train[i] = MSE(z_train,z_pred)

        mses.append(mse)
        mses_train.append(mse_train)
        labels.append('n = ' + str(deg))

    plot_lmb_MSE(np.log10(lambdas), mses, 'lasso', labels)
    plot_lmb_MSE(np.log10(lambdas), mses_train, 'lasso training', labels)


# calls
ols(8)
ridge(8)
# lasso(8)

# part g
# TODO: renaming the plotting files
terrain_datas = ['SRTM_data_Norway_1.tif', 'SRTM_data_Norway_2.tif']
x_size = 20
y_size = 20
for terrain_data in terrain_datas:

    xy,terrain = prep_terrain(terrain_data,x_size, y_size)

    # print(xy,terrain)
    # print(xy.shape)
    # print(terrain.shape)
    x_train, x_test, z_train, z_test = train_test_split(xy,terrain)

    ols(8)
    # ridge(8)
    # lasso(8)
