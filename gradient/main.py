import numpy as np
import matplotlib.pyplot as plt


def is_pos_def(A):
    print(f"lambda = {np.linalg.eigvals(A)}")
    return np.all(np.linalg.eigvals(A) > 0)


def exact_solution(A, b):
    return -np.linalg.inv(A).dot(b)


def function(x):
    return 0.5 * np.matmul(np.matmul(x.T, A), x) + np.matmul(b.T, x)


def diff(x):
    return np.matmul(1/2 * (np.add(A.T, A)), x) + b


def error(x):
    vect_sum = 0
    for elem in x:
        vect_sum += np.power(elem, 2)

    return np.sqrt(vect_sum)


def Gradient(A, x_0, eps):
    counter = 1
    x_prev = x_0
    x_next = x_prev - 1e-4 * diff(x_prev)

    func_sols = []

    while error(x_next - x_prev) > eps:
        counter += 1
        print(f"Функция: {function(x_prev)}, Градиент: {diff(x_prev)}, Норма: {error(x_next - x_prev)}")
        x_prev = x_next
        x_next = x_prev - 1e-4 * diff(x_prev)
        func_sols.append(function(x_prev))

    if is_pos_def(A):
        print("Матрица положительно определена")

    print("Точка минимума: {0}, значение функции: {1}".format(x_prev, function(x_prev)))
    print(f"Потребовалось итераций: {counter}, точность: {error(x_prev)}")

    return func_sols


if __name__ == "__main__":
    A = np.array([[13., 9., 8., 7.5, 3.5, 3.5],
                  [9., 10.5, 7., 10.5, 3.5, 4.],
                  [8., 7., 7.5, 8., 2., 5.5],
                  [7.5, 10.5, 8., 13.5, 3.5, 6.5],
                  [3.5, 3.5, 2., 3.5, 2.5, 1.5],
                  [3.5, 4., 5.5, 6.5, 1.5, 6.5]])

    b = np.array([1., 0., 0.5, 1.5, 1.5, 1.])
    x_0 = np.array([1.5, 0.5, 0.5, 0.5, 1., 1.])

    plt.plot(Gradient(A, x_0, 1e-5))
    print(f"Точное решение: {exact_solution(A, b)}")
    print(f"Значение функции в точке x*: {function(exact_solution(A, b))}")

    plt.grid()
    plt.xlabel("Iterations", fontdict={'family': 'serif', 'size': 15})
    plt.ylabel("f(x)", fontdict={'family': 'serif', 'size': 15})
    plt.show()
