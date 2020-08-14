from scarpa.generate.template import activate_template
from scarpa.generate.modulation import create_modulation
from scarpa.estimate.contraints import *
from scipy.signal import hilbert, correlate
from scipy.linalg import norm
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numpy import ndarray
from typing import Tuple, List
from math import floor, ceil


def vec2parms(x, p_len) -> Tuple[float, ndarray, ndarray]:
    mn = x[0]
    template = x[1 : 1 + p_len]
    am = x[1 + p_len :]
    return mn, template, am


def evaluate(template, actvect):
    model = activate_template(template, actvect)
    return model


def objective(
    x: ndarray, data: ndarray, activation_window: int, activation_indices: List[float]
) -> float:
    N: int = len(data)
    mn, template, am = vec2parms(x, activation_window)
    actvect = np.zeros(N)
    for i, a in zip(activation_indices, am):
        actvect[i] = a
    fit = evaluate(template, actvect)
    fit += mn
    residual = data - fit
    cost = norm(residual, 2)
    r2 = np.round(np.corrcoef(data, fit)[0, 1] ** 2, 2)
    print(f"R²={r2}, SS= {cost}")
    # cost = 1 - r2
    # print(cost)
    return cost


def cut_segments(data: ndarray, pre: int, post: int, indices: List[int]) -> ndarray:
    epochs = []
    for idx in indices:
        _ep = data[idx - pre : idx + pre]
        epochs.append(_ep)
    return np.atleast_2d(epochs).T


def deconvolve(
    data, activation_window: int, activation_indices: List[int]
) -> Tuple[float, ndarray, ndarray]:

    """estimate activation waveform and vector

    args
    ----
    data: ndarray
        the data to be fitted
    activation_window: int
        the length of a window for the activation template in samples
    activation_indices: List[int]
        the sample indices when an activation occurs
    
    returns
    -------
    ms:float
        the mean of the data
    template: ndarray
        the estimated activation waveform
    modulation: ndarray
        the estimated activation vector across the whole signal length

    """
    N: int = len(data)

    # estimate the initial template values
    segments = cut_segments(
        data,
        pre=floor(activation_window / 2),
        post=ceil(activation_window / 2),
        indices=activation_indices,
    )
    template = segments.mean(1)
    template -= template.mean(0)
    template /= norm(template)
    template = np.flip(template)
    # estimate the activation vector
    am = np.ones(len(activation_indices))
    am = template.dot(segments)

    # create the vector of initial values for minimize (argument there is x0)
    # count desc
    # 1     mean of the data vector
    # t     template
    initial = np.hstack((data.mean(), template, am))

    # initialize bounds
    # the template is bound between -1 and 1, and any amplitude is caused by
    # the am vector
    mb = (-np.inf, np.inf)
    tb = (-1.1, 1.1)
    ab = (-np.inf, np.inf)
    bnds = [mb] + [tb] * activation_window + [ab] * len(activation_indices)

    cons = [
        {"type": "eq", "fun": constraint_template_mean, "args": [activation_window]},
        {"type": "eq", "fun": constraint_template_range, "args": [activation_window]},
    ]

    solution = minimize(
        objective,
        args=(data, activation_window, activation_indices),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    mn_s, template_s, am = vec2parms(solution.x, p_len=activation_window)
    actvect = np.zeros(N)
    for i, a in zip(range(0, N, activation_window), am):
        actvect[i] = a
    print("Completed")
    return mn_s, template_s, actvect


def fit_with_plen_known(data, p_len: int) -> Tuple[float, ndarray, ndarray]:
    """estimate template and amplitude modulation with known period length

    args
    ----
    data: ndarray
        the data to be fitted
    p_len: int
        the length of a period of the template in samples
    
    returns
    -------
    ms:float
        the mean of the data
    template: ndarray
        the estimated template waveform of the periodic signal
    modulation: ndarray
        the estimated amplitude modulation across the whole signal length

    """
    N: int = len(data)
    p_count: float = N / p_len

    # estimate the initial template values
    template = data[: p_len * int(p_count)].reshape((int(p_count), p_len)).mean(0)
    template -= template.mean()
    template /= norm(template)
    template = np.flip(template)
    # estimate the activation vector
    am = np.ones((int(np.ceil(p_count))))

    # create the vector of initial values for minimize (argument there is x0)
    # count desc
    # 1     mean of the data vector
    # t     template
    initial = np.hstack((data.mean(), template, am))

    # initialize bounds
    # the template is bound between -1 and 1, and any amplitude is caused by
    # the am vector
    mb = (-np.inf, np.inf)
    tb = (-1.1, 1.1)
    ab = (-np.inf, np.inf)
    bnds = [mb] + [tb] * p_len + [ab] * len(am)

    cons = [
        {"type": "eq", "fun": constraint_template_mean, "args": [p_len]},
        {"type": "eq", "fun": constraint_template_range, "args": [p_len]},
    ]

    solution = minimize(
        objective,
        args=(data, p_len),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    mn_s, template_s, am = vec2parms(solution.x, p_len=100)
    actvect = np.zeros(N)
    for i, a in zip(range(0, N, p_len), am):
        actvect[i] = a
    return mn_s, template_s, actvect


if __name__ == "__main__":  # pragma: no cover

    from scarpa.generate.modulation import create_modulation
    from scarpa.generate.template import stack_template
    from scarpa.generate.shapes import sinus, hanning
    import matplotlib.pyplot as plt

    pcount = 23.5
    plen = 100
    samples = int(pcount * plen)
    anchors = hanning(23)

    template = sinus(plen)

    pure = stack_template(template, periodcount=pcount)
    modulation = create_modulation(anchors=anchors, samples=samples, kind="cubic")
    data = pure * modulation

    ac = 11
    mn_s, template_s, actvect = fit_with_plen_known(data, p_len=100)
    fit = activate_template(template_s, actvect)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(template, linewidth=5, color="gray")
    ax[0].plot(np.flip(template_s), linewidth=1, color="red")
    ax[1].plot(data, linewidth=5, color="gray")
    ax[1].plot(fit, linewidth=1, color="red")
    print(np.corrcoef(template_s, template)[0, 1])
