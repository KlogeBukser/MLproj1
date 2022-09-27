import numpy as np

from poly_funcs import get_2D_pols, get_2D_string, get_poly_index
from calculate import calc_design, find_beta, get_predict
from transform import bootstrap

class Model(object):
    """Regression model"""

    def __init__(self,polydeg):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        self.polydeg = polydeg
        self.features = get_poly_index(self.polydeg) + 1
        self.functions = get_2D_pols(self.polydeg)

    def get_beta(self):
        # Returns beta
        return self.beta

    def start_boot(self, X_train, z_train, n_boots, pred_boot = False):
        self.n_boots = n_boots
        self.boot_betas = np.empty((self.features, n_boots))

        if (pred_boot):
            # This option save the boot sample for z, and its prediction on X boot
            # Only use this option if you want these values
            z_boots = np.empty((len(z_train),self.n_boots))
            z_boots_fit = np.copy(z_boots)
            for i in range(n_boots):
                X_, z_ = bootstrap(X_train[:,:self.features],z_train)
                beta = find_beta(X_, z_)
                self.boot_betas[:,i] = beta
                z_boots[:,i] = z_
                z_boots_fit[:,i] = get_predict(X_, beta)

            return z_boots, z_boots_fit

        else:
            # Saves the beta values for each bootstrap sample
            for i in range(n_boots):
                X_, z_ = bootstrap(X_train[:,:self.features],z_train)
                self.boot_betas[:,i] = find_beta(X_, z_)

    def boot_predict(self,X):
        z_pred = np.empty((X.shape[0],self.n_boots))
        for i in range(self.n_boots):
            z_pred[:,i] = get_predict(X[:,:self.features],self.boot_betas[:,i])
        return z_pred

    def end_boot(self):
        self.n_boots = 0
        self.boot_betas = 0

    def design(self,x):
        # Uses the features to turn set of tuple values (x,y) into design matrix
        return calc_design(x, self.functions)

    def fit(self,X,z):
        # Finds coefficients beta (fitting the model) from input design matrix, and z
        # Scales design matrix down if model has reduced complexity
        self.beta = find_beta(X[:,:self.features],z)

    def predict(self,X):
        # Makes a prediction for z for the given design matrix
        return get_predict(X[:,:self.features],self.beta)

    def reduce_complexity(self):
        # Removes the features corresponding to the largest polynomial power
        # Resets beta as the fit is no longer valid
        # Return boolean describing whether reduction was successful

        if (self.polydeg == 0):
            # Stops program from reducing complexity when no longer possible
            return False

        self.end_boot()
        self.polydeg -= 1
        self.features = get_poly_index(self.polydeg) + 1
        self.beta = np.empty(self.features)
        self.functions = self.functions[:self.features]
        return True
