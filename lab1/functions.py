import numpy as np


def func1(a, x):
    return np.sin(a * x)


def func1dx(a, x):
    return a * np.cos(a * x)


def func1dx2(a, x):
    return (-1) * np.power(a, 2) * np.sin(a * x)


def func2(a, x):
    return a / (1 + 9 * np.power(x, 2))


def func2dx(a, x):
    return 18 * a * x / np.power(9 * np.power(x, 2) + 1, 2)


def func2dx2(a, x):
    return a * (648 * np.power(x, 2) / np.power(9 * np.power(x, 2) + 1, 3) - 18 / np.power(9 * np.power(x, 2) + 1, 2))