import numpy as np


class Solution:

    def __init__(self, r, a, y, sign):
        self.A = np.array([
            [0.9833079, 1.1372474, 0.9015589, 1.1052048],
            [0.8920470, 1.0466852, 0.8178930, 1.0121430],
            [1.0741852, 1.2570329, 0.9872952, 1.2199821],
            [1.0260374, 1.1960702, 0.9470140, 1.1691000]])
        self.b = np.array([
            [1.4173048],
            [1.5586898],
            [1.1403869],
            [1.1981015]])
        self.x_0 = np.array([
            [1.8007446],
            [1.9682616],
            [1.3134242],
            [1.6923226]])
        self.r = r
        self.a = a
        self.y = y
        self.sign = sign

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

    def newton(self, x_k: np.ndarray, epsilon=1e-6, max_iter=30):
        x_previous = x_k
        x_current = x_previous - np.linalg.inv(self.jacobian(x_previous[0:-1])).dot(self.lagrange(x_previous[0:-1]))
        iterator = 0
        while np.linalg.norm(x_current[0:-1] - x_previous[0:-1]) > epsilon and iterator < max_iter:
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
        print(f"\n||x*-x_0|| = {np.linalg.norm(x_star - self.x_0)}\n")

        for i in range(8):
            self.sign = -self.sign
            x_k = x_.copy()
            x_k[i // 2][0] += self.sign * self.a
            print(f"\nНачальное приближение {i + 1}:\n{x_k[0:-1]}")
            result = self.newton(x_k)
            print(f"Значение x: {result[0:-1]}")
            print(f"Значение y = {result[4][0]}")
            print(f"Значение функции = {self.f(result[0:-1])}\n")


if __name__ == "__main__":
    s_l = Solution(5, 4, 3, 1)
    s_l.start_lab()
