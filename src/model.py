import numpy as np

from poly_funcs import get_2D_pols, get_2D_string, get_poly_index
from calculate import calc_design, find_beta, get_predict

class Model(object):
    """Regression model"""

    def __init__(self,polydeg):
        # Collects the feature functions for a "2 variable polynomial of given degree"
        # Saves integers describing polynomial degree and number of features
        self.polydeg = polydeg
        self.features = get_poly_index(self.polydeg) + 1
        self.functions = get_2D_pols(self.polydeg)

    def get_polydeg(self):
        # Returns polynomial degree of model
        return self.polydeg

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

        self.polydeg -= 1
        self.features = get_poly_index(self.polydeg) + 1
        self.beta = np.empty(self.features)
        self.functions = self.functions[:self.features]
        return True
