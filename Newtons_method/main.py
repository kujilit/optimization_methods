import numpy as np


def error(x):
    vect_sum = 0
    for elem in x:
        vect_sum += np.power(elem, 2)

    return np.sqrt(vect_sum)


class Solution:

    def __init__(self, a, r, y, sign):
        self.A = np.array([
            [1.8835673, 1.7911470, 1.0321892, 1.0260374],
            [1.7911470, 1.0466852, 1.2432129, 1.1955702],
            [1.0321892, 1.2432129, 0.8567213, 1.9470140],
            [1.0260374, 1.1955702, 1.9470140, 1.1691000]])
        self.b = np.array([
            [1.2604948],
            [1.3466698],
            [1.1477929],
            [1.1215815]])
        self.x_0 = np.array([
            [1.9873446],
            [1.3245616],
            [1.3123642],
            [1.6912354]])
        self.r = r
        self.a = a
        self.y = y
        self.sign = sign

        print(np.linalg.det(self.A))

    def f(self, x: np.ndarray):
        result = .5 * x.T.dot(self.A).dot(x) + self.b.T.dot(x)
        return result[0][0]

    def lagrange(self, x: np.ndarray):
        return np.append((self.A + 2 * np.eye(4) * self.y).dot(x) + (self.b + 2 * self.y * self.x_0),
                         [[np.linalg.norm(x - self.x_0)**2 - self.r**2]], axis=0)

    def jacobian(self, x: np.ndarray):
        J_1_1 = self.A + 2 * np.eye(4) * self.y
        J_1_2 = 2 * (x - self.x_0)
        J_2_1 = J_1_2.T
        J_2_2 = [[0]]
        J_1 = np.append(J_1_1, J_1_2, axis=1)
        J_2 = np.append(J_2_1, J_2_2, axis=1)
        return np.append(J_1, J_2, axis=0)

    def newton(self, x_k, eps=1e-6, max_iter=30):
        x_previous = x_k
        x_current = x_previous - np.linalg.inv(self.jacobian(x_previous[0:-1])).dot(self.lagrange(x_previous[0:-1]))
        iterator = 0

        while error(x_current[0:-1] - x_previous[0:-1]) > eps and iterator < max_iter:
            iterator += 1
            x_previous = x_current
            x_current = x_previous - np.linalg.inv(self.jacobian(x_previous[0:-1])).dot(self.lagrange(x_previous[0:-1]))
        return x_current

    def start_lab(self):
        x_ = np.append(self.x_0, [[self.y]], axis=0)

        x_star = -np.linalg.inv(self.A).dot(self.b)
        f_in_x_star = self.f(x_star)
        print(f"x*:\n{x_star}")
        print(f"\nf(x*) = {f_in_x_star}")
        print(f"\nx*-x_0:\n{x_star - self.x_0}")
        print(f"\n||x*-x_0|| = {error(x_star - self.x_0)}\n")

        for i in range(8):
            print('======================')
            self.sign = -self.sign
            x_k = x_.copy()
            x_k[i // 2][0] += self.sign * self.a
            print(f"\n{i + 1}:\n{x_k[0:-1]} <-- Начальное приближение\n")
            result = self.newton(x_k)
            print(f"{result[0:-1]} <-- Значение x\n")
            print(f"{result[4][0]} <-- Значение y\n")
            print(f"{self.f(result[0:-1])} <-- Значение функции\n")


if __name__ == "__main__":
    Solution(a=1, r=2, y=3, sign=1).start_lab()
