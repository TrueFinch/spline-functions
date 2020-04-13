import numpy as np

from lab1.functions import func1, func1dx, func1dx2, func2, func2dx, func2dx2

# configurations for labs 1 and 2
configurations = [
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 1, "n": 1000, "x_start": 0, "x_end": np.pi},  # 0
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 2, "n": 1000, "x_start": 0, "x_end": np.pi},  # 1
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 3, "n": 1000, "x_start": 0, "x_end": np.pi},  # 2
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 4, "n": 1000, "x_start": 0, "x_end": np.pi},  # 3
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 5, "n": 1000, "x_start": 0, "x_end": np.pi},  # 4
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 6, "n": 1000, "x_start": 0, "x_end": np.pi},  # 5
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 7, "n": 1000, "x_start": 0, "x_end": np.pi},  # 6
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 8, "n": 1000, "x_start": 0, "x_end": np.pi},  # 7
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 9, "n": 1000, "x_start": 0, "x_end": np.pi},  # 8
    {"function": func1, "diff": func1dx, "diff2": func1dx2, "a": 10, "n": 1000, "x_start": 0, "x_end": np.pi},  # 9

    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 1, "n": 1000, "x_start": -1, "x_end": 1},  # 10
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 2, "n": 1000, "x_start": -1, "x_end": 1},  # 11
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 3, "n": 1000, "x_start": -1, "x_end": 1},  # 12
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 4, "n": 1000, "x_start": -1, "x_end": 1},  # 13
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 5, "n": 1000, "x_start": -1, "x_end": 1},  # 14
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 6, "n": 1000, "x_start": -1, "x_end": 1},  # 15
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 7, "n": 1000, "x_start": -1, "x_end": 1},  # 16
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 8, "n": 1000, "x_start": -1, "x_end": 1},  # 17
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 9, "n": 1000, "x_start": -1, "x_end": 1},  # 18
    {"function": func2, "diff": func2dx, "diff2": func2dx2, "a": 10, "n": 1000, "x_start": -1, "x_end": 1}  # 19
]
