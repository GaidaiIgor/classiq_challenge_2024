from scipy import optimize
import numpy as np


def calc_residuals(x, *args, **kwargs):
    calculated_precision = 10
    domain = np.arange(0, 1, 1 / 2 ** calculated_precision)
    prediction = x[0] + x[1] * domain
    actual = np.tanh(domain)
    return prediction - actual


def tanh_max_dev_poly1(x, domain):
    prediction = x[0] + x[1] * domain
    actual = np.tanh(domain)
    max_dev = np.max(abs(actual - prediction))
    return max_dev


if __name__ == '__main__':
    calculated_precision = 10
    num_segments = 2
    segment_length = 1 / num_segments
    for i in range(num_segments):
        domain = np.arange(i * segment_length, (i + 1) * segment_length, 1 / 2 ** calculated_precision)
        x0 = np.array([0, 0.5])
        res = optimize.minimize(tanh_max_dev_poly1, x0, domain, method='L-BFGS-B')
        print(f'Segment: {i}')
        print(res)
        print(res.x)
        print()
