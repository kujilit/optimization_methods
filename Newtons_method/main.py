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


def diff(x):
    return np.array([
        [a + 2 * np.eye(4) * y, 2 * (x - x_0)],
        [2 * (x - x_0).T, 0]
    ])

class Solution:
    def __init__(self, x_0: np.array, a: np.array, eps):
        self.a = a
        self.x_prev = x_0
        self.x_next = self.x_prev - np.linalg.inv(diff(self.x_prev)) * function(self.x_prev)

        while error(self.x_next - self.x_prev) <= eps:
            self.x_prev = self.x_next
            self.x_next = self.x_prev - np.linalg.inv(diff(self.x_prev)) * function(self.x_prev)


if __name__ == "__main__":
    a = np.array()
    b = np.array()
    y = 3
    x_0 = np.array()
    r = 5
