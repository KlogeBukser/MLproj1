import numpy as np

from poly_funcs import get_2D_pols, get_poly_index
from calculate import MSE, R2
from transform import bootstrap


class Model(object):
    """Regression model"""

    def __init__(self,polydeg,x_train,z_train, train_name = "train", regression_method='ols'):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        # Takes training data, saves z_train the design matrix X_train
        self.polydeg = polydeg
        self.feature_count = get_poly_index(self.polydeg) + 1
        self.functions = get_2D_pols(self.polydeg)
        self.datapoints = len(z_train)
        self.z_train = z_train
        self.X_dict = {train_name:self.design(x_train)}
        self.beta = self.choose_beta(self.X_dict["train"], self.z_train, regression_method)


    def find_beta_ols(self,X,y):
        ''' Finds beta using OLS '''
        square = np.dot(X.T,X)                      #nf*nf
        if np.linalg.det(square):
            inv = np.linalg.inv(square)             #nf*nf

        else:
            # psuedoinversion for singular matrices
            inv = np.linalg.pinv(square)

        beta = np.dot(np.dot(inv,X.T),y)        #nf*1

        return beta


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
                'For single lambda use interger for lamb_range'

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

    def choose_beta(self, X, z, regression_method='ols'):
        '''return the beta found using the specified regression method
        input: X: matrix, z: array like '''
        if regression_method == 'ols':
            beta = self.find_beta_ols(X, z)
        elif regression_method == 'ridge':
            beta = self.best_ridge_beta(X, z)
        else:
            raise Exception('Invalid regression_method, try something else you bastard!')

        return beta


    def start_boot(self, n_boots, predict_boot = False, regression_method='ols'):
        self.n_boots = n_boots
        self.boot_betas = np.empty((self.feature_count, n_boots))

        if (predict_boot):
            # This option returns the boot sample for z, and its prediction on X boot
            # Only use this option if you want these values
            z_boots = np.empty((self.datapoints,self.n_boots))
            z_boots_fit = np.copy(z_boots)
            for i in range(n_boots):
                X_, z_ = bootstrap(self.X_dict["train"],self.z_train)

                beta = self.choose_beta(X_, z_, regression_method)

                self.boot_betas[:,i] = beta
                z_boots[:,i] = z_
                z_boots_fit[:,i] = np.dot(X_, beta)

            return z_boots, z_boots_fit

        else:
            # Saves the beta values for each bootstrap sample
            for i in range(n_boots):
                X_, z_ = bootstrap(self.X_dict["train"],self.z_train)
                self.boot_betas[:,i] = self.choose_beta(X_, z_,regression_method)

    def boot_predict(self,name):
        X = self.X_dict[name]
        z_pred = np.empty((X.shape[0],self.n_boots))
        for i in range(self.n_boots):
            z_pred[:,i] = np.dot(X,self.boot_betas[:,i])
        return z_pred

    def end_boot(self):
        self.n_boots = 0
        self.boot_betas = 0

    def design(self,x):
        # Uses the features to turn set of tuple values (x,y) into design matrix

        n = x.shape[0]
        n_funcs = len(self.functions)
        design = np.ones((n, n_funcs))

        for i in range(n):
            for j in range(n_funcs):
                design[i,j] = self.functions[j](x[i])

        return design

    def predict(self,name):
        # Makes a prediction for z for the given design matrix
        X = self.X_dict[name]
        return np.dot(X,self.beta)

    def reduce_complexity(self):
        # Removes the features corresponding to the largest polynomial power
        # Resets beta as the fit is no longer valid
        # Return boolean describing whether reduction was successful

        if (self.polydeg == 0):
            # Stops program from reducing complexity when no longer possible
            return False

        self.end_boot()
        self.polydeg -= 1
        self.feature_count = get_poly_index(self.polydeg) + 1

        self.functions = self.functions[:self.feature_count]
        for name, mat in self.X_dict.items():
            self.X_dict[name] = mat[:,:self.feature_count]
        self.beta = self.find_beta_ols(self.X_dict["train"],self.z_train)
        return True
