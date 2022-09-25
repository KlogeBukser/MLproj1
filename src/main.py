import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from produce_results import *

np.random.seed(2022)

x, z = generate_data_Franke(20,noise = 0.5)

plot_MSE(x,z,8)
