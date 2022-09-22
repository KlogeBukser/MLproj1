import numpy as np

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def generate_data_noise():
    # Makes meshgrid of x,y in the range between 0 and 1
    # returns a vector z with an added normal distribution with z_0 = 0, sigma = 1
    x = np.arrange(0,1,0.05)
    y = np.arrange(0,1,0.05)
    x,y = np.meshgrid(x,y)
    return = FrankeFunction(x,y) + np.random.normal()
