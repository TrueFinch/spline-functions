import numpy as np
import matplotlib.pyplot as plt

from lab1.utils import h, calc_matrix
from lab1.configurations import configurations
from scipy import integrate


def integral(i, x, y, m):
    return h(x, i + 1) * (y[i + 1] + y[i]) / 2 + np.power(h(x, i + 1), 2) * (m[i] - m[i + 1]) / 12


def main(configuration):
    func = configuration["function"]
    diff = configuration["diff"]
    diff2 = configuration["diff2"]
    a = configuration["a"]
    n = configuration["n"]
    x = np.linspace(configuration["x_start"], configuration["x_end"], num=n + 1)
    y = np.array([func(a, x_) for x_ in x])

    m_res = calc_matrix(n, x, y, diff2, a)

    y_res = []
    s_res = []

    f = lambda x: func(a, x)

    for i in range(n):
        y_integral_val = integrate.quad(f, x[i], x[i + 1])
        y_res.append(y_integral_val[0])
        s_res.append(integral(i, x, y, m_res))

    d = sum([abs(y_res[i] - s_res[i]) for i in range(0, len(s_res))]) / len(s_res)
    print(d)

    return 0


if __name__ == "__main__":
    main(configurations[0])
