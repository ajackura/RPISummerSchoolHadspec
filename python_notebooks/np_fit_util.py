import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# convert number to string at desired precision
def num_to_str(number, precision = 4):
    return "%0.*e" % (precision, number)

# computes parameter errors from covariance matrix
def param_errors(covariance):
    return np.sqrt(np.diag(covariance))

# computes correlation matrix from covariance
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# pretty print a matrix
def pretty_print_matrix(a):
    for i in a:
        for j in i:
            print(num_to_str(j), end=' ')
        print('')
    return
