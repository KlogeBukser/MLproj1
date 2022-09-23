import numpy as np

def get_2D_pols(deg):
    # Produces every polynomial of two variables up to a given degree
    # Functions take a tuple as input
    # i = i and j = j are necessary to prevent lambda functions from overwriting themselves
    pols = []
    for j in range(deg + 1):
        # Outer loop iterates over polynomial degrees
        # Inner loop iterates over polynomials of the degree in the outer loop
        pols += [(lambda xy, i = i, j = j: xy[0]**(j - i) * xy[1]**i) for i in range(j + 1)]
    return pols
