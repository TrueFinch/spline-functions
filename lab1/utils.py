import numpy as np


def h(x, i):
    return x[i + 1] - x[i] if i + 1 < len(x) else x[i] - x[i - 1]


def m(x, i):
    return h(x, i - 1) / (h(x, i + 1) + h(x, i))


def l(x, i):
    return h(x, i) / (h(x, i + 1) + h(x, i))


# returns list of answer for matrix
def calc_matrix(n, x, y, diff2, a):
    m_a = np.zeros((n + 1) ** 2).reshape(n + 1, n + 1)
    # m_a[0][0] = 2
    m_a[0][0] = (y[1] - y[0]) / h(x, 0)
    m_a[0][1] = 1
    for i in range(1, n):
        m_a[i][i - 1] = m(x, i)
        m_a[i][i] = 2
        m_a[i][i + 1] = l(x, i)
    m_a[n][n - 1] = 1
    m_a[n][n] = (y[n] - y[n - 1]) / h(x, 0)
    # m_a[n][n] = 2

    m_b = np.zeros(n + 1)
    m_b[0] = 3 * (y[1] - y[0]) / h(x, 1) - h(x, 1) * diff2(a, x[0]) / 2
    for i in range(1, n):
        m_b[i] = 3 * (m(x, i) * (y[i + 1] - y[i]) / h(x, i + 1) + l(x, i) * (y[i] - y[i - 1]) / h(x, i))
    m_b[n] = 3 * ((y[n] - y[n - 1]) / h(x, n)) + h(x, n) * diff2(a, x[n]) / 2

    m_res = np.linalg.solve(m_a, m_b)
    return m_res
