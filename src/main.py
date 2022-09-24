import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate import *
from calculate import *
from poly_funcs import *
from transform import *
from plotting import *
from produce_results import *

x,z = generate_data_Franke()

prod_score_plots(x, z, max_poly = 5, include_R2 = True)
