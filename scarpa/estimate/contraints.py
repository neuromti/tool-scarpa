import numpy as np


def constraint_template_mean(x, p_len):
    "constrain the template to always have zero mean"
    template = x[1 : 1 + p_len]
    return np.mean(template)


def constraint_template_range(x, p_len):
    "constrain the template to always have range 2"
    template = x[1 : 1 + p_len]
    return np.ptp(template) - 2
