from scarpa.generate.template import activate_template
from scarpa.generate.modulation import create_modulation
from scipy.signal import hilbert, correlate
from scipy.linalg import norm
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numpy import ndarray
from typing import Tuple


def vec2parms(x, p_len) -> Tuple[float, ndarray, ndarray]:
    mn = x[0]
    template = x[1 : 1 + p_len]
    am = x[1 + p_len :]
    return mn, template, am


def constraint_template_mean(x, p_len):
    "constrain the template to always have zero mean"
    template = x[1 : 1 + p_len]
    return np.mean(template)


def constraint_template_range(x, p_len):
    "constrain the template to always have range 2"
    template = x[1 : 1 + p_len]
    return np.ptp(template) - 2


def evaluate(template, actvect):
    model = activate_template(template, actvect)
    return model


def objective(x: ndarray, data: ndarray, p_len: int) -> float:
    N: int = len(data)
    mn, template, am = vec2parms(x, p_len)
    actvect = np.zeros(N)
    for i, a in zip(range(0, N, p_len), am):
        actvect[i] = a
    fit = evaluate(template, actvect)
    fit += mn
    residual = data - fit
    cost = norm(residual, 2)
    print(cost)
    return cost


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
    from scarpa.generate.template import stack_periodic
    from scarpa.generate.shapes import sinus, hanning
    import matplotlib.pyplot as plt

    pcount = 23.5
    plen = 100
    samples = int(pcount * plen)
    anchors = hanning(23)

    template = sinus(plen)

    pure = stack_periodic(template, periodcount=pcount)
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
