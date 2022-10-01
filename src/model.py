import numpy as np

from poly_funcs import get_2D_pols
from calculate import MSE, R2

class Algorithms:

    def __init__(self, regression_method):
        self.REGRESSION_METHODS = ['ols','ridge']

        self.regression_method = regression_method


        self.assign_algos()


    def check_health(self):

        if self.regression_method not in self.REGRESSION_METHODS:
            raise Exception('Invalid regression_method')


    def assign_algos(self):

        self.check_health()
        self.assign_reg()


    def assign_reg(self):
        """assign the correct regression_method to self.resample


        """
        if self.regression_method == 'ols':
            self.find_beta = self.find_beta_ols

        if self.regression_method == 'ridge':
            self.find_beta = self.best_ridge_beta


    def find_beta_ols(self, X, y):
        ''' Finds beta using OLS '''
        return self.find_beta_ridge(X, y, 0)


    def find_beta_ridge(self, X, y, lamb):

        '''return the beta for the given lambda '''

        sqr = X.T @ X
        dim = sqr.shape[0]
        mat = sqr-lamb*np.identity(dim)

        if np.linalg.det(mat):
            inv = np.linalg.inv(mat)             #nf*nf

        else:
            # psuedoinversion for singular matrices
            inv = np.linalg.pinv(mat)

        beta = inv @ X.T @ y

        return beta

    def best_ridge_beta(self, X, y, nlambdas=100, lamb_range=(-4,4), min_func=MSE):

        if nlambdas == 1:

            try:
                lamb = float(lamb_range)
            except:
                'For single lambda use integer for lamb_range'

            return self.find_beta_ridge(X, y, lamb_range)


        lambdas = np.logspace(lamb_range[0], lamb_range[1], nlambdas)
        best_beta = self.find_beta_ridge(X, y, lambdas[0])
        self.best_lamb = lambdas[0]

        for lamb in lambdas:
            beta = self.find_beta_ridge(X, y, lamb)
            if self.cmp_beta(X, y, beta, best_beta, cmp_func=min_func):
                best_beta = beta
                self.best_lamb = lamb

        return best_beta

    def cmp_beta(self, X, y, beta_1, beta_2, cmp_func):
        '''return a true if beta_1 has smaller cpm_func and false otherwise '''

        if cmp_func(y, X @ beta_1) < cmp_func(y, X @ beta_2): # ?questionable comparison
            return True
        return False


class Model:
    """Regression model"""

    def __init__(self, polydeg, regression_method='ols'):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        # Takes training data, saves z_train the design matrix X_train
        self.polydeg = polydeg
        self.functions = get_2D_pols(self.polydeg)
        self.feature_count = len(self.functions)

        # NEW, unstable
        self.algorithms = Algorithms(regression_method)

    def bootstrap(self, x_test, n_boots):
        """ Predicts on test, and training set

        :z_test: array_like
        :n_boots: int
        :returns: 2D array_like, 2D array_like

        """
        X_train = self.X_train
        X_test = self.design(x_test)
        z_pred = np.empty((X_test.shape[0],n_boots))
        z_fit = np.empty((X_train.shape[0],n_boots))

        for i in range(n_boots):
            X_, z_ = self.one_boot(X_train,self.z_train)
            beta = self.algorithms.find_beta(X_, z_).ravel()
            z_pred[:,i] = np.dot(X_test, beta)
            z_fit[:,i] = np.dot(X_train, beta)

        return z_pred, z_fit

    def one_boot(self, X, z):
        X_ = np.empty(X.shape)
        z_ = np.empty(z.shape)
        n_z = z_.shape[0]
        for s in range(n_z):
            r_int = np.random.randint(n_z)
            X_[s,:] = X[r_int,:]
            z_[s] = z[r_int]
        return X_, z_

    def fit(self, x, z):
        """ Fits the model on training data x, and z

        :x: array_like of tuples
        :z: array_like

        """
        self.z_train = z
        self.X_train = self.design(x)
        self.beta = self.algorithms.find_beta(self.X_train,self.z_train)

    def design(self,x):
        # Uses the features to turn set of tuple values (x,y) into design matrix
        n = x.shape[0]
        design = np.ones((n, self.feature_count))
        for i in range(n):
            for j in range(self.feature_count):
                design[i,j] = self.functions[j](x[i])

        return design

    def predict_mat(self,X):
        return X @ self.beta

    def predict(self,x):
        # Makes a prediction for z for the given design matrix
        return self.predict_mat(self.design(x))


    def cross_validate(self, k, n_lambs = 100, lamb_range = (-4,4), score_func = MSE):
        z = self.z_train
        X = self.X_train

        n_z = z.shape[0]
        order = np.arange(n_z)

        gen = np.random.default_rng()
        gen.shuffle(order)

        lambdas = np.logspace(lamb_range[0],lamb_range[1],n_lambs)
        fold_len = int(n_z/k)
        folds = [order[i*fold_len:(i+1)*fold_len] for i in range(k - 1)]
        folds.append(order[(k-1)*fold_len:])

        scores = np.empty((lambdas.shape[0],k))

        for i in range(k):
            test_indices = folds.pop(0)
            train_indices = np.array(folds).ravel()

            z_test = z[test_indices]
            X_test = X[test_indices]

            z_train = z[train_indices]
            X_train = X[train_indices]

            for j in range(n_lambs):
                beta = self.algorithms.find_beta_ridge(X_train, z_train, lambdas[j])
                z_pred = X_test @ beta
                scores[j,i] = score_func(z_test,z_pred)

            # Adding the test fold to the back of the folds
            folds.append(test_indices)
        return scores
