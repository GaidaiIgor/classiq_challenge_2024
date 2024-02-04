from scipy import optimize
import numpy as np


def calc_residuals(x, *args, **kwargs):
    calculated_precision = 10
    domain = np.arange(0, 1, 1 / 2 ** calculated_precision)
    prediction = x[0] + x[1] * domain
    actual = np.tanh(domain)
    return prediction - actual


if __name__ == '__main__':
    x0 = [0.5, 0.5]
    res = optimize.least_squares(calc_residuals, x0)
    print(res)
    print(res.x)
