import numpy as np

# Polynomials
# Naming convention is (poly)(variables)(exponent for each variable)
# This should be possible to generalise
def get_2D_pols(deg):
    """This thing makes every 2D polynomial for up to a given degree"""
    poll_list = [[(lambda x, i = i, j = j : x[0]**(j - i) * x[1]**i) for i in range(j + 1)] for j in range(deg + 1)]
    pols = []
    for elem_list in poll_list:
        for elem in elem_list:
            pols.append(elem)
    return pols
