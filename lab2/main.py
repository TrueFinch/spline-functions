import numpy as np
import math
import matplotlib.pyplot as plt

from lab1.configurations import configurations
from scipy import integrate


class Lab2:
    def __init__(self, points, diff2):
        self.points = points
        self.n = len(points) - 1
        self.diff2 = diff2
        self.m_res = []
        self.__calc_matrix__()

    def __get_x__(self, i):
        _, x_, _ = self.points[i]
        return x_

    def __get_y__(self, i):
        _, _, y_ = self.points[i]
        return y_

    def __get_eps__(self, i):
        eps_, _, _ = self.points[i]
        return eps_

    def calc_i(self, x):
        for i in range(len(self.points) - 1):
            x1 = self.__get_x__(i)
            x2 = self.__get_x__(i + 1)
            if x1 <= x <= x2:
                return i
        assert False, "ERROR: 'i' not found!"

    def __h__(self, i):
        if i + 1 < len(self.points):
            return self.__get_x__(i + 1) - self.__get_x__(i)
        else:
            return self.__get_x__(i) - self.__get_x__(i - 1)

    def __calc_matrix__(self):
        m_a = np.zeros((self.n + 1) ** 2).reshape(self.n + 1, self.n + 1)
        m_a[0][0] = 2
        m_a[0][1] = 1
        for i in range(1, self.n):
            m_a[i][i - 1] = self.__mu__(i) * (1 - math.pow(self.__get_eps__(i), 2) / math.pow(self.__h__(i - 1), 2))
            m_a[i][i] = 2 + math.pow(self.__get_eps__(i), 2) / (self.__h__(i - 1) * self.__h__(i))
            m_a[i][i + 1] = self.__lmbd__(i) * (1 - math.pow(self.__get_eps__(i), 2) / math.pow(self.__h__(i), 2))
        m_a[self.n][self.n - 1] = 1
        m_a[self.n][self.n] = 2

        m_b = np.zeros(self.n + 1)
        term1 = ((self.__get_y__(1) - self.__get_y__(0)) / self.__h__(0) - self.diff2(self.__get_x__(0)))
        m_b[0] = 6 / self.__h__(0) * term1
        for i in range(1, self.n):
            term1 = 6 / (self.__h__(i - 1) + self.__h__(i))
            term2 = (self.__get_y__(i + 1) - self.__get_y__(i)) / self.__h__(i)
            term3 = (self.__get_y__(i) - self.__get_y__(i - 1)) / self.__h__(i - 1)
            m_b[i] = term1 * (term2 - term3)

        term1 = 3 * ((self.__get_y__(self.n) - self.__get_y__(self.n - 1)) / self.__h__(self.n))
        term2 = self.__h__(self.n) * self.diff2(self.__get_x__(self.n)) / 2
        m_b[self.n] = term1 + term2

        self.m_res = np.linalg.solve(m_a, m_b)

    def sdx(self):
        term1 = 0.5 * sum([self.__h__(j) * (self.__get_y__(j) + self.__get_y__(j + 1)) for j in range(0, self.n - 1)])
        ss = [math.pow(self.__h__(j), 3) * (self.m_res[j] + self.m_res[j + 1]) for j in range(0, self.n - 1)]
        term2 = 1 / 24 * sum(ss)
        return term1 - term2

    def __mu__(self, i):
        return self.__h__(i - 1) / (self.__h__(i - 1) + self.__h__(i))

    def __lmbd__(self, i):
        return 1 - self.__mu__(i)


def main(configuration):
    func = configuration["function"]
    diff = configuration["diff"]
    diff2 = configuration["diff2"]
    a = configuration["a"]
    n = configuration["n"]
    x = np.linspace(configuration["x_start"], configuration["x_end"], num=n + 1)
    y = np.array([func(a, x_) for x_ in x])
    e = np.linspace(1e-5, 1, num=100)
    d = []
    for e_ in e:
        eps = [e_ for i in range(n + 1)]
        solver = Lab2(list(zip(eps, x, y)), lambda x_: diff2(a, x_))

        y_res = integrate.quad(lambda x_: func(a, x_), configuration["x_start"], configuration["x_end"])[0]
        s_res = solver.sdx()

        d_ = abs(y_res - s_res)
        # print("eps {} : error {}".format(e_, d_))
        d.append(d_)
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(e, d, label="error")
    ax.legend()
    ax.set_xlabel('eps')
    ax.set_ylabel('error')
    plt.show()
    return 0


if __name__ == "__main__":
    for i in [0, 2, 5, 9, 10, 12, 15, 19]:
        print("a = {}".format(i % 10 + 1))
        main(configurations[i])
