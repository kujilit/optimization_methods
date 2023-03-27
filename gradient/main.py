import numpy as np
import matplotlib.pyplot as plt


def is_pos_def(A):
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


def Gradient(A, x_0, steps, eps):
    counter = 1
    x_prev = x_0
    x_next = x_prev - 1e-4 * diff(x_prev)

    func_sols = []

    if steps == 0 and is_pos_def(A):
        print("Матрица положительно определена")
        print(f"lambda = {np.linalg.eigvals(A)}")
        while error(x_next - x_prev) > eps:
            counter += 1
            x_prev = x_next
            x_next = x_prev - 1e-4 * diff(x_prev)
            func_sols.append(function(x_prev))

    elif is_pos_def(A):
        while counter < steps:
            counter += 1
            x_prev = x_next
            x_next = x_prev - 1e-4 * diff(x_prev)
            func_sols.append(function(x_prev))

    else:
        print("Матрица не определена положительно")

    print("\nТочка минимума: {0}, значение функции: {1}".format(x_next, function(x_prev)))
    print(f"Потребовалось итераций: {counter}, точность: {error(x_next - x_prev)}")

    return func_sols, counter, x_next


if __name__ == "__main__":
    A = np.array([[13., 9., 8., 7.5, 3.5, 3.5],
                  [9., 10.5, 7., 10.5, 3.5, 4.],
                  [8., 7., 7.5, 8., 2., 5.5],
                  [7.5, 10.5, 8., 13.5, 3.5, 6.5],
                  [3.5, 3.5, 2., 3.5, 2.5, 1.5],
                  [3.5, 4., 5.5, 6.5, 1.5, 6.5]])

    b = np.array([1., 0., 0.5, 1.5, 1.5, 1.])
    x_0 = np.array([1.5, 0.5, 0.5, 0.5, 1., 1.])

    func_solves, steps, x = Gradient(A, x_0, 0, 1e-7)
    plt.plot(func_solves)
    print(f"Результат для 1/4 итераций: {Gradient(A, x_0, round(steps/4), 1e-7)[2]}\n")
    print(f"Результат для 1/3 итераций: {Gradient(A, x_0, round(steps / 3), 1e-7)[2]}\n")
    print(f"Результат для 1/2 итераций: {Gradient(A, x_0, round(steps / 2), 1e-7)[2]}\n")

    print(f"Точное решение: {exact_solution(A, b)}")
    print(f"Значение функции в точке x*: {function(exact_solution(A, b))}")

    plt.grid()
    plt.xlabel("Iterations", fontdict={'family': 'serif', 'size': 15})
    plt.ylabel("f(x)", fontdict={'family': 'serif', 'size': 15})
    plt.savefig('chart.png', bbox_inches='tight')
    plt.show()
