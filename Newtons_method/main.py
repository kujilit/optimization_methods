import numpy as np


def function(x):
    return 0.5 * np.matmul(np.matmul(x.T, a), x) + np.matmul(b.T, x)


def error(x):
    vect_sum = 0
    for elem in x:
        vect_sum += np.power(elem, 2)

    return np.sqrt(vect_sum)


def lagrange(x):
    return function(x) + y * (pow(error(x - x_0), 2) - pow(r, 2))


def lagrange_diff(x):
    return np.matmul(a, x) + b + 2 * y * (x - x_0)


if __name__ == "__main__":
    a = np.array()
    b = np.array()
    y = 3
    x_0 = np.array()
    r = 5
