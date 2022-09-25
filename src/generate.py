import numpy as np

def FrankeFunction(x,y):
    # Finds z from x,y

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



def generate_data_Franke(n = 20, noise = 1):
    # Produces feature values
    # Gives noise to label values

    x = np.zeros((n**2,2))
    vals = np.arange(0,1,1/n)
    for i in range(n):
        x[i*n:(i+1)*n,0] = vals
        x[i::n,1] = vals

    z = FrankeFunction(x[:,0],x[:,1]) + noise*np.random.randn(x.shape[0])

    return x, z
