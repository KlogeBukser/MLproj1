import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample


from poly_funcs import get_2D_pols
from model import *
from calculate import *
from plotting import *

from generate import *
from produce_results import *

np.random.seed(1)

n_boots = 20
n = 20


x, z = generate_data_Franke(n, noise = 0.8)
x_train, x_test, z_train, z_test = train_test_split(x,z)
is_terrain = False



def ols(polynomial_degree = 5):


    # Makes models for each polynomial degree, and feeds them the testing data (x_test) for predictions

    # puts plot in another the terrain plots files
    if is_terrain:
        file_dir = 'terrain plots'
    else:
        file_dir = 'plots'

    n_pol = polynomial_degree + 1 # since the first polynomial degree is 0


    poly_degs = np.arange(n_pol)
    test_score, train_score, bias, var, Kfold_score = np.empty((5,n_pol))

    n_simple = np.min((6,n_pol))
    simple_degs = np.arange(n_simple)
    simple_MSE, simple_R2 = np.empty((2,n_simple))
    betas = []

    for deg in poly_degs:

        # trainning model for each polynomial degree

        model = Model(deg)
        model.train(x_train,z_train)
        z_pred_train = model.predict(x_train)

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
    plot_beta(n,betas, file_dir=file_dir)
    # Plots MSE + R2
    plot_simple_scores(n,simple_MSE,simple_R2, file_dir=file_dir)

    # Plots KFold, bias/variance, test/train comparison
    plot_boot_scores(n,poly_degs,test_score,train_score,bias,var,Kfold_score, file_dir=file_dir)


def ridge(polynomial_degree = 5):

    if is_terrain:
        file_dir = 'terrain plots'
    else:
        file_dir = 'plots'

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)
    ridge_score, ols_score,k_ridge_score, k_ols_score, k_score_lam = np.empty((5,n_pol))


    n_lambdas = 20
    lambdas = np.logspace(-4,4,n_lambdas)
    polydeg_lam = [0,1,2,5,6,7,8]
    test_score_lam = np.empty((len(polydeg_lam),n_lambdas))
    k_score_lam = np.empty((len(polydeg_lam),n_lambdas))
    deg_lamb = 0


    for deg in poly_degs:

        model = Ridge(deg)
        model.train(x_train,z_train,'best')

        z_pred, z_fit = model.bootstrap(x_test,n_boots)
        ridge_score[deg] = MSE(z_test,z_pred)
        k_ridge_score[deg] = model.cross_validate(6)

        model.set_lambda(0)
        z_pred, z_fit = model.bootstrap(x_test,n_boots)
        ols_score[deg] = MSE(z_test,z_pred)
        k_ols_score[deg] = model.cross_validate(6)


        if deg in polydeg_lam:

            for i,lamb in enumerate(lambdas):
                model.set_lambda(lamb)
                pred, fit = model.bootstrap(x_test,n_boots)
                test_score_lam[deg_lamb,i] = MSE(z_test,pred)
                k_score_lam[deg_lamb,i] = model.cross_validate(6)

            deg_lamb += 1



    plot_2D(np.log10(lambdas), test_score_lam, plot_count = len(test_score_lam), label = ['p = ' + str(i) for i in polydeg_lam],
        title='Ridge Test-MSE ' + str(n**2) + ' points',x_title='$log10(\lambda)$',y_title='Error',filename= 'Ridge MSE-lambda.pdf', multi_x=False, file_dir=file_dir)

    plot_2D(np.log10(lambdas), k_score_lam, plot_count = len(k_score_lam), label = ['p = ' + str(i) for i in polydeg_lam],
        title='Ridge F_fold-MSE ' + str(n**2) + ' points',x_title='$log10(\lambda)$',y_title='Error',filename= 'Ridge K_fold-lambda.pdf', multi_x=False, file_dir=file_dir)


    plot_2D(poly_degs, [ridge_score,ols_score,k_ridge_score,k_ols_score], plot_count = 4, label = ['Ridge','OLS','Ridge predict','OLS predict'],
        title='Ridge + OLS comparison with predictions using K-fold method' + ' OLS comparison ' + str(n**2) + ' points',x_title='polynomial degree',y_title='Error',filename= 'Ridge OLS compare Kfold.pdf', multi_x=False, file_dir=file_dir)




def lasso(polynomial_degree = 5):


    if is_terrain:
        file_dir = 'terrain plots'
    else:
        file_dir = 'plots'

    n_pol = polynomial_degree + 1
    poly_degs = np.arange(n_pol)

    nlambdas = 100
    lambdas = np.logspace(-4, 4, nlambdas)

    # for plotting
    mses_k = []
    mses_train_k = []
    mses_b = []
    mses_train_b = []
    labels = []

    # # bootstrap
    # bs = cross_validation.Bootstrap(9, random_state=0)

    # cross validation
    k = 5
    kfold = KFold(n_splits = k)

    n_boostraps = 20

    for deg in poly_degs:

        poly = PolynomialFeatures(degree = deg)

        X = poly.fit_transform(x_train)
        X_test = poly.fit_transform(x_test)

        # will scale data with StandardScaler

        mse_k = np.zeros(nlambdas)
        mse_train_k = np.zeros(nlambdas)

        mse_avg_b = np.zeros(nlambdas)
        mse_train_b = np.zeros(nlambdas)


        for i in range(nlambdas):
            lmb = lambdas[i]

            reg_lasso = linear_model.Lasso(lmb)
            # make_pipeline(StandardScaler(with_mean=False), linear_model.Lasso(lmb))

            # bootstrap
            mse_boot = np.zeros(n_boostraps)
            for j in range(n_boostraps):
                x_, z_ = resample(x_train, z_train)
                X_ = poly.fit_transform(x_)
                reg_lasso.fit(X_,z_)
                
                z_boot_pred = reg_lasso.predict(X_test)
                mse_boot[j] = MSE(z_test, z_boot_pred)

            mse_avg_b[i] = np.mean(mse_boot)

            # cross val

            reg_lasso.fit(X,z_train)
            cvs_train = cross_val_score(reg_lasso, X, z_train, scoring='neg_mean_squared_error', cv=kfold)
            mse_train_k[i] = np.mean(-cvs_train)

            cvs_test = cross_val_score(reg_lasso, X_test, z_test, scoring='neg_mean_squared_error', cv=kfold)
            mse_k[i] = np.mean(-cvs_test)

        mses_b.append(mse_avg_b)

        mses_k.append(mse_k)
        mses_train_k.append(mse_train_k)
        labels.append('n = ' + str(deg))


    # only plotting every second polynomial degree because there are too many
    # bootstrap
    plot_lmb_MSE(np.log10(lambdas), mses_b[::2], 'lasso with bootstrap', labels[::2], filename="lasso bootstrap.pdf", file_dir=file_dir)

    # k-fold
    # plot_lmb_MSE(np.log10(lambdas), mses_train_k, 'lasso on train with k-fold', labels)
    plot_lmb_MSE(np.log10(lambdas), mses_k[::2], 'lasso with k-fold', labels[::2], filename="lasso k-fold.pdf", file_dir=file_dir)
    


# calls
# ols(8)
ridge(8)
lasso(8)

# part g
is_terrain = True

terrain_datas = ['SRTM_data_Norway_1.tif']

# we want to cut out part of data as the data file contains way too many points
x_size = n
y_size = n

for terrain_data in terrain_datas:

    xy,terrain = prep_terrain(terrain_data,x_size, y_size)

    x_train, x_test, z_train, z_test = train_test_split(xy,terrain)

    # ols(8)
    # ridge(8)
    # lasso(6)
