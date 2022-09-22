import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from calculate import *
from poly_funcs import *

x,z = generate_data_Franke()

poly_deg = np.arange(1,6,1)
R2_vals = np.zeros(5)
for deg in poly_deg:
    funcs = get_2D_pols(deg)
    X = calc_design(x,funcs)
    beta = find_coeffs(X,z)
    z_ = get_model(X,beta)
    R2_vals[deg-1] = R2(z,z_)


plt.plot(poly_deg,R2_vals)
plt.show()
