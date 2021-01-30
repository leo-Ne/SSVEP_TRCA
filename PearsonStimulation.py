from typing import Any, Union

import numpy as np


template = np.arange(0, 250, 1)
template_std = np.std(template) * np.sqrt(250)
print(template_std)


def PearsonCorrealtion(buffer: np.ndarray, length):
    accumulator = 0
    x_avg       = 0
    for i in range(length):
        x_avg       += buffer[i]

    x_avg   = x_avg / length
    x_std   = 0
    for i in range(length):
        dep         =  buffer[i] - x_avg            # Depolarization
        accumulator += dep * template[i]            # Covariance
        x_std       += dep ** 2
    x_std   = np.sqrt(x_std)
    print(x_std)
    std_dev = x_std * template_std
    correlation = accumulator / std_dev
    print("x_std:\t", x_std)
    print("tempalte_std:\t", template_std)
    print("std_dev:\t", std_dev)
    print("accumulator:\t", accumulator)
    print("accumulator / std_dev:\t", correlation)
    pass


def avg_vector(x:np.ndarray):
    length = x.size
    x_avg = 0
    for i in range(length):
        x_avg = x[i]
    x_avg = x_avg / length
    return x_avg


def pearsonCorr(x1:np.ndarray, x2:np.ndarray) -> float:
    length = x1.size
    # x1_avg = 0
    # x2_avg = 0
    x1_avg = avg_vector(x1)
    x2_avg = avg_vector(x2)

    x1_std  = 0
    x2_std  = 0
    x1x2_cov = 0
    """
    The computation in depolarization and  covariance could be done in the same step.
    """
    for i in range(length):
        x1_dep = x1[i] - x1_avg
        x2_dep = x2[i] - x2_avg
        x1_std += x1_dep ** 2            # without divided by N.
        x2_std += x2_dep ** 2
        x1x2_cov += x1_dep * x2_dep
    x1_std = np.sqrt(x1_std)
    x2_std = np.sqrt(x2_std)
    std_dev = x1_std * x2_std
    pearsonCorr = x1x2_cov / std_dev
    return pearsonCorr


if __name__ == "__main__":
    # PearsonCorrealtion(template, 250)
    template1 = template
    template2 = template * -1
    template3 = template * 0.5
    p1 = pearsonCorr(template, template1)
    p2 = pearsonCorr(template, template2)
    p3 = pearsonCorr(template, template3)
    print("test pearsonCorr")
    print(p1, p2, p3)
    pass
