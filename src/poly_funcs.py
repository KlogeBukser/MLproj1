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

def get_2D_string(deg):
    # Method for collecting the polynomials as strings
    str_pols = []
    for j in range(deg + 1):
        str_pols += [get_poly_text(i,j) for i in range(j + 1)]
    return str_pols

def get_poly_index(deg):
    # Returns the index of the last polynomial of a given degree
    index = 0
    for i in range(deg):
        index += i + 2
    return index

def get_poly_text(i,j):
    # Helper function for getting the string version of a specific polynomial
    # i is the degree of y, while j is the total polynomial degree
    if (i == j == 0):
        return "$1$"
    poly = "$"
    if (j-i > 0):
        poly += "x"
    if (j-i > 1):
        poly += "^" + str(j-i)
    if (i > 0):
        poly += "y"
    if (i > 1):
        poly += "^" + str(i)
    return poly + "$"
