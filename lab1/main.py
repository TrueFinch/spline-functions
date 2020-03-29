import numpy as np
import matplotlib.pyplot as plt

from lab1.configurations import configurations


def h(x, i):
    return x[i + 1] - x[i] if i + 1 < len(x) else x[i] - x[i - 1]


def m(x, i):
    return h(x, i - 1) / (h(x, i + 1) + h(x, i))


def l(x, i):
    return h(x, i) / (h(x, i + 1) + h(x, i))


def t(x, x_, i):
    return (x_ - x[i]) / h(x, i)


def s(x_, x, m_, y_, n_, diff, a):
    for i in range(0, n_):
        if x[i] <= x_ <= x[i + 1]:
            a_ = 6 * ((y_[i + 1] - y_[i]) / h(x, i + 1) - (m_[i + 1] + 2 * m_[i]) / 3) / h(x, i + 1)
            b_ = 12 * ((m_[i + 1] + m_[i]) / 2 - (y_[i + 1] - y_[i]) / h(x, i + 1)) / np.power(h(x, i + 1), 2)
            return y_[i] + diff(a, x_) * (x_ - x[i]) + a_ * np.power(x_ - x[i], 2) / 2 + b_ * np.power(x_ - x[i], 3) / 6


def main(configuration):
    func = configuration["function"]
    diff = configuration["diff"]
    diff2 = configuration["diff2"]
    a = configuration["a"]
    n = configuration["n"]
    x = np.linspace(configuration["x_start"], configuration["x_end"], num=n + 1)
    y = np.array([func(a, x_) for x_ in x])

    m_a = np.zeros((n + 1) ** 2).reshape(n + 1, n + 1)
    # m_a[0][0] = 2
    m_a[0][0] = (y[1] - y[0]) / h(x, 0)
    m_a[0][1] = 1
    for i in range(1, n):
        m_a[i][i - 1] = m(x, i)
        m_a[i][i] = 2
        m_a[i][i + 1] = l(x, i)
    m_a[n][n - 1] = 1
    m_a[n][n] = 2
    # m_a[n][n] = (y[n] - y[n - 1]) / h(x, 0)

    m_b = np.zeros(n + 1)
    m_b[0] = 3 * (y[1] - y[0]) / h(x, 1) - h(x, 1) * diff2(a, x[0]) / 2
    for i in range(1, n):
        m_b[i] = 3 * (m(x, i) * (y[i + 1] - y[i]) / h(x, i + 1) + l(x, i) * (y[i] - y[i - 1]) / h(x, i))
    m_b[n] = 3 * ((y[n] - y[n - 1]) / h(x, n)) + h(x, n) * diff2(a, x[n]) / 2

    m_res = np.linalg.solve(m_a, m_b)

    x_dots = np.linspace(configuration["x_start"], configuration["x_end"], num=10000)
    y_dots = func(a, x_dots)
    s_dots = np.array([s(x_, x, m_res, y, n, diff, a) for x_ in x_dots])
    plt.plot(x_dots, y_dots, label="func")
    plt.plot(x_dots, s_dots, label="spline")
    plt.legend()
    plt.show()

    d = sum([abs(y_dots[i] - s_dots[i]) for i in range(0, len(y_dots))]) / len(y_dots)
    print(d)
    return 0


if __name__ == "__main__":
    main(configurations[10])
