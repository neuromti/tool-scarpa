from scarpa.generate.template import stack_periodic
from scarpa.generate.modulation import create_modulation
from scipy.signal import hilbert, correlate
from scipy.linalg import norm
import numpy as np
from scipy.optimize import minimize
from numpy import ndarray


def vec2parms(x, p_len, am_count):
    mn = x[0]
    template = x[1 : 1 + p_len]
    anchors = x[-am_count:]
    return mn, template, anchors


def constraint_template_mean(x, p_len):
    "constrain the template to always have zero mean"
    template = x[1 : 1 + p_len]
    return np.mean(template)


def evaluate(template, p_count, am_anchors, am_interp):
    model = stack_periodic(template, p_count)
    am = create_modulation(am_anchors, samples=N, kind=am_interp)
    fit = model * am
    return fit


def objective(x: ndarray, data: ndarray, p_len: int, am_count: int, am_interp: str):
    N: int = len(data)
    p_count: float = N / p_len

    mn, template, am_anchors = vec2parms(x, p_len, am_count)
    fit = evaluate(template, p_count, am_anchors, am_interp)
    fit += mn
    residual = data - fit
    cost = norm(residual, 2)
    print(cost)
    return cost


def fit_with_plen_known(
    data, p_len: int, am_count: int, am_interp: str = "nearest",
):
    N: int = len(data)
    p_count: float = N / p_len

    # estimate the initial template values
    template = data[: p_len * int(p_count)].reshape((int(p_count), p_len)).mean(0)
    template -= template.mean()
    template /= norm(template)

    # estimate the initial amplitude modulation values
    am = abs(hilbert(data))
    am = [am[int(idx)] for idx in np.linspace(0, len(data), am_count, endpoint=False)]

    # create the vector of initial values for minimize (argument there is x0)
    # count desc
    # 1     mean of the data vector
    # a     anchor_values
    # t     template
    initial = np.ones(1 + p_len + am_count)
    initial[0] = data.mean()
    initial[1 : p_len + 1] = template
    initial[-am_count:] = am

    # initialize bounds
    # the template is bound between -1 and 1, and any amplitude is caused by
    # the am vector
    mb = (-np.inf, np.inf)
    tb = (-1, 1)
    ab = (min(data), max(data))
    bnds = [mb] + [tb] * p_len + [ab] * am_count

    cons = [
        {"type": "eq", "fun": constraint_template_mean, "args": [p_len]},
    ]

    solution = minimize(
        objective,
        args=(data, p_len, am_count, am_interp),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )

    best_parms = vec2parms(solution.x, p_len, am_count)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(best_parms[1])
    ax[1].plot(best_parms[2])

