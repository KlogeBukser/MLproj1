import numpy as np

from poly_funcs import get_2D_pols
from calculate import MSE, R2

class Algorithms:

    def __init__(self, regression_method, resampling_method):
        self.REGERSSION_METHODS = ['ols','ridge']
        self.RESAMPLING_METHODS = ['none','boot','cross']

        self.regression_method = regression_method
        self.resampling_method = resampling_method

        self.assign_algos()


    def check_health(self):

        if self.regression_method not in self.REGERSSION_METHODS:
            raise Exception('Invalid regression_method')

        if self.resampling_method not in self.RESAMPLING_METHODS:
            raise Exception('Invalid resampling_method')

    def assign_algos(self):

        self.check_health()

        self.assign_reg()
        # self.assign_resample()


    def assign_reg(self):
        """assign the correct regression_method to self.resample

        :resampling_method: string

        """
        if self.resampling_method == 'boot':
            pass
            # self.resample = self.bootstrap

        if self.resampling_method == 'cross':
            pass
            # self.resample = self.cross


    def assign_resample(self):
        """assign the correct resampling_method to self.resample

        :resampling_method: string

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

        sqr = np.dot(X.T,X)
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

        # print(self.best_lamb)
        return best_beta

    def cmp_beta(self, X, y, beta_1, beta_2, cmp_func):
        '''return a true if beta_1 has smaller cpm_func and false otherwise '''

        if cmp_func(y, X @ beta_1) < cmp_func(y, X @ beta_2): # ?questionable comparison
            return True
        return False


    def start_boot(self, n_boots, regression_method='ols', predict_boot = False):
        self.n_boots = n_boots
        self.boot_betas = np.empty((self.feature_count, n_boots))

        if (predict_boot):
            # This option returns the boot sample for z, and its prediction on X boot
            # Only use this option if you want these values
            z_boots = np.empty((len(self.z_train),self.n_boots))
            z_boots_fit = np.copy(z_boots)
            for i in range(n_boots):
                X_, z_ = bootstrap(self.X_dict["train"],self.z_train)

                beta = self.find_beta(X_, z_, regression_method)

                self.boot_betas[:,i] = beta
                z_boots[:,i] = z_
                z_boots_fit[:,i] = np.dot(X_, beta)

            return z_boots, z_boots_fit

        else:
            # Saves the beta values for each bootstrap sample
            for i in range(n_boots):
                X_, z_ = bootstrap(self.X_dict["train"],self.z_train)
                self.boot_betas[:,i] = self.find_beta(X_, z_,regression_method)

    def boot_predict(self,name):
        X = self.X_dict[name]
        z_pred = np.empty((X.shape[0],self.n_boots))
        for i in range(self.n_boots):
            z_pred[:,i] = np.dot(X,self.boot_betas[:,i])
        return z_pred

    def end_boot(self):
        self.n_boots = None
        self.boot_betas = None


class Model:
    """Regression model"""

    def __init__(self,polydeg,x_train,z_train, train_name = "train", regression_method='ols', resampling_method='none'):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        # Takes training data, saves z_train the design matrix X_train
        self.polydeg = polydeg
        self.functions = get_2D_pols(self.polydeg)
        self.feature_count = len(self.functions)
        self.z_train = z_train
        self.X_dict = {train_name:self.design(x_train)}
        self.reg_method = regression_method
        self.res_method = resampling_method
        self.choose_beta()
        self.beta = self.find_beta(self.X_dict["train"],self.z_train)


        # NEW, unstable
        self.algorithms = Algorithms(regression_method, resampling_method)


    def find_beta_ols(self, X, y):
        ''' Finds beta using OLS '''
        return self.find_beta_ridge(X, y, 0)


    def find_beta_ridge(self, X, y, lamb):

        '''return the best for the best lambda for the given lambda'''

        sqr = np.dot(X.T,X)
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

        # print(self.best_lamb)
        return best_beta

    def cmp_beta(self, X, y, beta_1, beta_2, cmp_func):
        '''return a true if beta_1 has smaller cpm_func and false otherwise '''

        if cmp_func(y, X @ beta_1) < cmp_func(y, X @ beta_2): # ?questionable comparison
            return True
        return False


    def add_x(self,x,name):
        self.X_dict[name] = self.design(x)

    def choose_beta(self):
        '''return the beta found using the specified regression method
        input: X: matrix, z: array like '''
        if self.reg_method == 'ols':
            self.find_beta = self.find_beta_ols
        elif self.reg_method == 'ridge':
            self.find_beta = self.best_ridge_beta
        else:
            raise Exception('Invalid regression_method, try something else you bastard!')


    def boot_resample(self):
        X_train = self.X_dict["train"]
        z_train = self.z_train
        X_ = np.empty(X_train.shape)
        z_ = np.empty(z_train.shape)
        n_z = len(z_train)
        for s in range(n_z):
            r_int = np.random.randint(n_z)
            X_[s,:] = X_train[r_int,:]
            z_[s] = z_train[r_int]
        return X_, z_

    def start_boot(self, n_boots, predict_boot = False):
        self.n_boots = n_boots
        self.boot_betas = np.empty((self.feature_count, n_boots))

        if (predict_boot):
            # This option returns the boot sample for z, and its prediction on X boot
            # Only use this option if you want these values
            z_boots = np.empty((len(self.z_train),self.n_boots))
            z_boots_fit = np.copy(z_boots)

            for i in range(n_boots):

                X_, z_ = self.boot_resample()

                beta = self.find_beta(X_, z_)

                self.boot_betas[:,i] = beta
                z_boots[:,i] = z_
                z_boots_fit[:,i] = np.dot(X_, beta)

            return z_boots, z_boots_fit

        else:
            # Saves the beta values for each bootstrap sample
            for i in range(n_boots):
                X_, z_ = self.boot_resample()
                self.boot_betas[:,i] = self.find_beta(X_, z_)

    def boot_predict(self,name):
        X = self.X_dict[name]
        z_pred = np.empty((X.shape[0],self.n_boots))
        for i in range(self.n_boots):
            z_pred[:,i] = np.dot(X,self.boot_betas[:,i])
        return z_pred

    def end_boot(self):
        self.n_boots = None
        self.boot_betas = None

    def design(self,x):
        # Uses the features to turn set of tuple values (x,y) into design matrix

        n = x.shape[0]
        design = np.ones((n, self.feature_count))

        for i in range(n):
            for j in range(self.feature_count):
                design[i,j] = self.functions[j](x[i])

        return design

    def predict(self,name):
        # Makes a prediction for z for the given design matrix
        X = self.X_dict[name]
        return np.dot(X,self.beta)
