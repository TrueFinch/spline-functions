import numpy as np
import matplotlib.pyplot as plt

from lab1.configurations import configurations
from lab1.utils import h, calc_matrix


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

    m_res = calc_matrix(n, x, y, diff2, a)

    x_dots = np.linspace(configuration["x_start"], configuration["x_end"], num=10000)
    y_dots = func(a, x_dots)
    s_dots = np.array([s(x_, x, m_res, y, n, diff, a) for x_ in x_dots])
    plt.plot(x_dots, y_dots, label="func")
    plt.plot(x_dots, s_dots, label="spline")
    plt.legend()
    plt.show()

    d = sum([abs(y_dots[i] - s_dots[i]) for i in range(0, len(y_dots))]) / len(y_dots)
    print(np.sqrt(d))
    return 0


if __name__ == "__main__":
    main(configurations[0])
