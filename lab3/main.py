import numpy as np
import matplotlib.pyplot as plt
import math


def func_x(a, b, t) -> float:
    return a * np.exp(b * t) * np.cos(t)
    # return 5 * np.tan(t)


def func_y(a, b, t) -> float:
    return a * np.exp(b * t) * np.sin(t)
    # return 5 * np.power(np.cos(t), 2)


class Lab3:
    def __init__(self, p, q, points):
        self.p = p
        self.q = q
        self.points = points
        self.n = len(points) - 1
        self.is_x = True
        self.m_res = []

        n = self.n

        d = lambda i: self.__calc_dist__(i)
        c = lambda i: self.__calc_c_term__(i)
        P = lambda i: self.__calc_p__(i)
        Q = lambda i: self.__calc_q__(i)

        self.s = np.zeros(n + 1)
        for i in range(n + 1):
            self.s[i] = sum([d(j) for j in range(0, i - 1)])

        y0 = d(0) / d(1)
        yn = d(n - 1) / d(n - 2)

        term1 = 2 * (self.__get_node__(1) - self.__get_node__(0)) / d(0)
        term2 = 2 * np.power(y0, 2)
        term3 = (self.__get_node__(2) - self.__get_node__(1)) / d(1)
        c1_star = term1 - term2 * term3

        term4 = 2 * (self.__get_node__(n) - self.__get_node__(n - 1)) / d(n - 1)
        term5 = 2 * np.power(yn, 2)
        term6 = (self.__get_node__(n - 1) - self.__get_node__(n - 2)) / d(n - 2)
        c2_star = term4 - term5 * term6  # c_(n-1)*

        m_a = np.zeros((n + 1) ** 2).reshape(n + 1, n + 1)
        m_a[0][0] = 1
        m_a[0][1] = -(np.power(y0, 2) - 1)
        m_a[0][2] = -np.power(y0, 2)

        term7 = self.__calc_lmbd__(1) * P(0) * (1 + np.power(y0, 2) + self.__get_q__(0))
        term8 = self.__calc_mu__(1) * Q(1) * (2 + self.__get_p__(1))
        m_a[1][1] = term7 + term8
        m_a[1][2] = self.__calc_mu__(1) * Q(1) + self.__calc_lmbd__(1) * P(0) * np.power(y0, 2)

        for i in range(2, n - 1):
            m_a[i][i - 1] = self.__calc_lmbd__(i) * P(i - 1)
            iterm1 = self.__calc_lmbd__(i) * P(i - 1) * (2 + self.__get_q__(i - 1))
            iterm2 = self.__calc_mu__(i) * Q(i) * (2 + self.__get_p__(i))
            m_a[i][i] = iterm1 + iterm2
            m_a[i][i + 1] = self.__calc_mu__(i) * Q(i)

        m_a[n - 1][n - 2] = self.__calc_lmbd__(n - 1) * P(n - 2) + self.__calc_mu__(n - 1) * np.power(yn, 2) * Q(n - 1)

        term13 = self.__calc_lmbd__(n - 1) * P(n - 2) * (2 + self.__get_q__(n - 2))
        term14 = self.__calc_mu__(n - 1) * Q(n - 1) * (1 + np.power(yn, 2) + self.__get_p__(n - 1))
        m_a[n - 1][n - 1] = term13 + term14

        m_a[n][n - 2] = -np.power(yn, 2)
        m_a[n][n - 1] = -(np.power(yn, 2) - 1)
        m_a[n][n] = 1

        m_b = np.zeros(n + 1)
        m_b[0] = c1_star
        m_b[1] = c(1) - self.__calc_lmbd__(1) * P(0) * c1_star
        for i in range(2, n - 1):
            m_b[i] = c(i)
        m_b[n - 1] = c(n - 1) - self.__calc_mu__(n - 1) * Q(n - 1) * c2_star
        m_b[n] = c2_star

        self.m_res = np.linalg.solve(m_a, m_b)

    def s_x(self, t):
        return self.__s__(t, True)

    def s_y(self, t):
        return self.__s__(t, False)

    def __calc_a__(self, i):
        return self.__get_node__(i + 1) - self.__calc_c__(i)

    def __calc_b__(self, i):
        return self.__get_node__(i) - self.__calc_d__(i)

    def __calc_c__(self, i):
        term1 = (3 + self.__get_q__(i)) * (self.__get_node__(i + 1) - self.__get_node__(i))
        term2 = self.__calc_dist__(i) * self.__get_m__(i)
        term3 = (2 + self.__get_q__(i)) * self.__calc_dist__(i) * self.__get_m__(i + 1)
        term4 = (2 + self.__get_q__(i)) * (2 + self.__get_p__(i)) - 1
        return (-term1 + term2 + term3) / term4

    def __calc_c_term__(self, i):
        term1 = self.__calc_lmbd__(i) * self.__calc_p__(i - 1) * (3 + self.__get_q__(i - 1))
        term2 = (self.__get_node__(i) - self.__get_node__(i - 1)) / self.__calc_dist__(i - 1)
        term3 = self.__calc_mu__(i) * self.__calc_q__(i) * (3 + self.__get_p__(i))
        term4 = (self.__get_node__(i + 1) - self.__get_node__(i)) / self.__calc_dist__(i)
        return term1 * term2 + term3 * term4

    def __calc_d__(self, i):
        term1 = (3 + self.__get_p__(i)) * (self.__get_node__(i + 1) - self.__get_node__(i))
        term2 = self.__calc_dist__(i) * self.__get_m__(i + 1)
        term3 = (2 + self.__get_p__(i)) * self.__calc_dist__(i) * self.__get_m__(i)
        term4 = (2 + self.__get_q__(i)) * (2 + self.__get_p__(i)) - 1
        return (term1 - term2 - term3) / term4

    def __calc_dist__(self, i):
        return np.sqrt(np.power(self.__get_x__(i + 1) - self.__get_x__(i), 2) +
                       np.power(self.__get_y__(i + 1) - self.__get_y__(i), 2))

    def __calc_i__(self, s):
        for i in range(0, len(self.s) - 1):
            s1 = self.s[i]
            s2 = self.s[i + 1]
            if s1 <= s <= s2:
                # print(i)
                return i
        assert False, "ERROR: 'i' not found!"

    def __calc_lmbd__(self, i):
        return self.__calc_dist__(i) / (self.__calc_dist__(i - 1) + self.__calc_dist__(i))

    def __calc_mu__(self, i):
        return 1 - self.__calc_lmbd__(i)

    def __calc_p__(self, i):
        term1 = 3 + 3 * self.__get_p__(i) + math.pow(self.__get_p__(i), 2)
        term2 = (2 + self.__get_q__(i)) * (2 + self.__get_p__(i)) - 1
        return term1 / term2

    def __calc_q__(self, i):
        term1 = 3 + 3 * self.__get_q__(i) + math.pow(self.__get_q__(i), 2)
        term2 = (2 + self.__get_q__(i)) * (2 + self.__get_p__(i)) - 1
        return term1 / term2

    def __get_m__(self, i):
        return self.m_res[i]

    def __get_p__(self, i):
        return self.p[i]

    def __get_q__(self, i):
        return self.q[i]

    def __get_node__(self, i):
        _, x_, y_ = self.points[i]
        return x_ if self.is_x else y_

    def __get_x__(self, i):
        _, x_, _ = self.points[i]
        return x_

    def __get_y__(self, i):
        _, _, y_ = self.points[i]
        return y_

    def __s__(self, s, is_x: bool):
        self.is_x = is_x
        i = self.__calc_i__(s)
        t = (s - self.s[i]) / self.__calc_dist__(i)
        # t = s
        # print(t)
        term1 = self.__calc_a__(i) * t
        term2 = self.__calc_b__(i) * (1 - t)
        term3 = (self.__calc_c__(i) * np.power(t, 3)) / (1 + self.__get_p__(i) * (1 - t))
        term4 = (self.__calc_d__(i) * np.power(1 - t, 3)) / (1 + self.__get_q__(i) * t)
        return term1 + term2 + term3 + term4
        # term1 = self.__get_node__(i) * (1 - t)
        # term2 = self.__get_node__(i + 1) * t
        # term3 = self.__calc_c__(i) * (np.power(t, 3) / (1 + self.__get_p__(i) * (1 - t)) - t)
        # term4 = self.__calc_d__(i) * (np.power(1 - t, 3) / (1 + self.__get_q__(i) * t) - (1 - t))
        # return term1 + term2 + term3 + term4


def main():
    n = 100
    t1 = -3 * np.pi
    t2 = 3 * np.pi
    a_param = 0.01
    b_param = 0.15
    t = np.linspace(t1, t2, num=n + 1)
    x = func_x(a_param, b_param, t)
    y = func_y(a_param, b_param, t)

    p = [-0.5 for i in range(0, n)]
    q = [-0.5 for i in range(0, n)]

    solver = Lab3(p, q, list(zip(t, x, y)))

    n2 = 1000
    tt = np.linspace(t1, t2, num=n2)
    f_x_dots = np.array([func_x(a_param, b_param, t_) for t_ in tt])
    f_y_dots = np.array([func_y(a_param, b_param, t_) for t_ in tt])
    s_x_dots = np.array([solver.s_x(s_) for s_ in np.linspace(0, solver.s[len(solver.s) - 1], num=n2)])
    s_y_dots = np.array([solver.s_y(s_) for s_ in np.linspace(0, solver.s[len(solver.s) - 1], num=n2)])
    sum = 0
    for i in range(n2):
        sum = math.pow(f_x_dots[i] - s_x_dots[i], 2) + math.pow(f_y_dots[i] - s_y_dots[i], 2)
    sum = math.sqrt(sum) / n2
    print(sum)

    plt.plot(f_x_dots, f_y_dots, label="func", lw=3)
    plt.plot(s_x_dots, s_y_dots, label="spline")
    plt.legend()
    # plt.xlim(1, -1)
    # plt.ylim(1, -1)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
