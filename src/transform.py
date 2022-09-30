from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_scale(X,z):

    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size = 0.2)
    scaler = StandardScaler()
    scaler.fit(X_train[:,1:])
    X_train[:,1:] = scaler.transform(X_train[:,1:])
    X_test[:,1:] = scaler.transform(X_test[:,1:])

    return X_train, X_test, z_train, z_test


def cross_valid(X,z): # not finished

    n = len(z)
    X_ = np.empty(X.shape)
    z_ = np.empty(n)

    return X_, z_
