from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    # res[:, int(shape[1]/2):shape[1]] = -0.1
    res[shape[0]-200:shape[0], 0:80] = 0.1
    res[shape[0]-200:shape[0], 80:int(shape[1]/2)+5] = 1
    # res[shape[0]-200:shape[0], int(shape[1]/2):shape[1]] = -1
    # res[0:shape[0]-300, int(shape[1]/2):shape[1]] = 0.2
    # res[300:, 200:] = 1
    # ---
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    # res[:, 0:int(shape[1]/2)] = -0.1
    res[shape[0]-200:shape[0], int(shape[1]/2):shape[1]-80] = 1
    res[shape[0]-200:shape[0], shape[1]-80:shape[1]] = 0.1
    # res[shape[0]-200:shape[0], 0:int(shape[1]/2)] = -1
    # res[0:shape[0]-300, 0:int(shape[1]/2)] = 0.2
    # ---
    return res
