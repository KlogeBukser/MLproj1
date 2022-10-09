import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.misc

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
    z = z[:,np.newaxis]

    return x, z

def prep_terrain(tif_name, x_size, y_size):
    '''Format terrain data

    tif_name: string, file name
    size: 2-tuple, the number of point interested for x and y
    return: xy: 2d array, all combination of x and y 
    z: array, 
    '''

    # Load the terrain
    terrain = imageio.imread(tif_name)

    assert 0 < x_size <= terrain.shape[1], 'invalid shape'
    assert 0 < y_size <= terrain.shape[0], 'invalid shape' 

    # print("x=0,y=2", terrain[2][0])
    # caution: column major, index (y,x)
    x = np.arange(x_size)
    y = np.arange(y_size)

    xy = np.array(np.meshgrid(y,x)).T.reshape(-1, 2)
    z = terrain.flatten()[:x_size*y_size]
    z = z[:,np.newaxis]

    return xy,z

