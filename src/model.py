import numpy as np

from poly_funcs import get_2D_pols
from calculate import MSE, R2


class Model:
    """Regression model"""

    def __init__(self, polydeg):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        # Takes training data, saves z_train the design matrix X_train

        self.polydeg = polydeg
        self.functions = get_2D_pols(self.polydeg)
        self.feature_count = len(self.functions)
        self.is_scaled = False

    def train(self,x,z):
        self.z_train = z
        self.X_train = self.design(x)
        self.beta = self.find_beta(self.X_train,self.z_train)

    def find_beta(self, X, y):
        ''' Finds beta using OLS '''
        sqr = X.T @ X

        if np.linalg.det(sqr):
            inv = np.linalg.inv(sqr)             #nf*nf

        else:
            # psuedoinversion for singular matrices
            inv = np.linalg.pinv(sqr)

        beta = inv @ X.T @ y

        return beta

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
            beta = self.find_beta(X_, z_).ravel()
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


    def design(self,x):
        # Uses the features to turn set of tuple values (x,y) into design matrix
        n = x.shape[0]
        design = np.ones((n, self.feature_count))
        for i in range(n):
            for j in range(self.feature_count):
                design[i,j] = self.functions[j](x[i])

        if  not self.is_scaled:
            self.design_mean, self.design_std = self.calc_scale(design)
            is_scaled = True

        return design#(design - self.design_mean)*self.design_std

    def calc_scale(self,X):
        mean = np.array([0] + [np.mean(X[:,i]) for i in range(1,self.feature_count)])
        std = np.array([1] + [1/np.std(X[:,i]) for i in range(1,self.feature_count)])
        return mean, std


    def predict(self,x):
        # Makes a prediction for z for the given design matrix
        return self.design(x) @ self.beta


    def cross_validate(self, k):
        z = self.z_train
        X = self.X_train

        n_z = z.shape[0]
        order = np.arange(n_z)

        gen = np.random.default_rng()
        gen.shuffle(order)

        scores = np.empty(k)

        fold_len = int(n_z/k)
        folds = [order[i*fold_len:(i+1)*fold_len] for i in range(k - 1)]
        folds.append(order[(k-1)*fold_len:])


        for i in range(k):
            test_indices = folds.pop(0)
            train_indices = [elem for fold in folds for elem in fold]

            z_test = z[test_indices]
            X_test = X[test_indices]

            z_train = z[train_indices]
            X_train = X[train_indices]

            K_mean, K_std = self.calc_scale(X_train)
            X_train = K_std*(X_train - K_mean)
            X_test = K_std*(X_test - K_mean)

            # Adding the test fold to the back of the folds
            folds.append(test_indices)


            beta = self.find_beta(X_train, z_train)
            z_pred = X_test @ beta
            scores[i] = MSE(z_test,z_pred)


        return np.mean(scores)


class Ridge(Model):
    def __init__(self, polydeg):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        # Takes training data, saves z_train the design matrix X_train
        super().__init__(polydeg)


    def train(self,x,z,lamb = 'best'):

        self.z_train = z
        self.X_train = self.design(x)
        self.set_lambda(lamb)


    def set_lambda(self,lamb = 'best'):
        if (lamb == 'best'):
            self.best_beta()
            return

        self.lamb = lamb
        self.beta = self.find_beta(self.X_train, self.z_train)


    def find_beta(self, X, z,lamb = 'self'):

        '''return the beta for the given lambda '''

        if (lamb == 'self'):
            lamb = self.lamb

        sqr = X.T @ X
        dim = sqr.shape[0]

        mat = sqr + lamb*np.eye(dim,dim)

        if False:#np.linalg.det(mat):
            inv = np.linalg.inv(mat)

        else:
            # psuedoinversion for singular matrices
            inv = np.linalg.pinv(mat)

        beta = inv @ X.T @ z

        return beta


    def best_beta(self, nlambdas=100, lamb_range=(-4,4)):

        X = self.X_train
        z = self.z_train

        if nlambdas == 1:

            try:
                lamb = float(lamb_range)
            except:
                'For single lambda use integer for lamb_range'

            return self.find_beta(X, z, lamb_range)

        lambdas = np.logspace(lamb_range[0], lamb_range[1], nlambdas)
        best_beta = self.find_beta(X, z, lambdas[0])
        best_lamb = lambdas[0]

        score = 10
        for lamb in lambdas:
            beta = self.find_beta(X, z, lamb)
            if MSE(z,X @ beta) < score:#self.cmp_beta(X, z, beta, best_beta):
                best_beta = beta
                best_lamb = lamb
                score = MSE(z,X @ beta)


        self.lamb = best_lamb
        self.beta = best_beta
