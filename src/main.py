import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from calculate import *
from poly_funcs import *

x,z = generate_data_Franke()
funcs = [poly200,poly210,poly201,poly220,poly202,poly211]
X = calc_design(x,funcs)
beta = find_coeffs(X,z)
print(beta)
