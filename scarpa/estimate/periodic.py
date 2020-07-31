from scarpa.generate.template import stack_periodic
from scarpa.generate.modulation import create_modulation
from scipy.signal import hilbert, correlate
from scipy.linalg import norm
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numpy import ndarray
from typing import Tuple


def vec2parms(x, p_len, am_count) -> Tuple[float, ndarray, ndarray]:
    mn = x[0]
    template = x[1 : 1 + p_len]
    anchors = x[-am_count:]
    return mn, template, anchors


def constraint_template_mean(x, p_len):
    "constrain the template to always have zero mean"
    template = x[1 : 1 + p_len]
    return np.mean(template)


def constraint_template_range(x, p_len):
    "constrain the template to always have range 2"
    template = x[1 : 1 + p_len]
    return np.ptp(template) - 2


def evaluate(template, p_count, am_anchors, am_interp):
    N = int(p_count * len(template))
    model = stack_periodic(template, p_count)
    am = create_modulation(am_anchors, samples=N, kind=am_interp)
    fit = model * am
    return fit


def objective(
    x: ndarray, data: ndarray, p_len: int, am_count: int, am_interp: str
) -> float:
    N: int = len(data)
    p_count: float = N / p_len

    mn, template, am_anchors = vec2parms(x, p_len, am_count)
    fit = evaluate(template, p_count, am_anchors, am_interp)
    fit += mn
    residual = data - fit
    cost = norm(residual, 2)
    # print(cost)
    return cost


def fit_with_plen_known(
    data, p_len: int, am_count: int, am_interp: str = "nearest",
) -> Tuple[float, ndarray, ndarray]:
    """estimate template and amplitude modulation with known period length

    args
    ----
    data: ndarray
        the data to be fitted
    p_len: int
        the length of a period of the template in samples
    am_count: int
        the number of anchor points to estimate amplitude modulation
    am_interp:str
        the kind of interpolation between amplitude anchorpoints to cover the whole data
    
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
        {"type": "eq", "fun": constraint_template_range, "args": [p_len]},
    ]

    solution = minimize(
        objective,
        args=(data, p_len, am_count, am_interp),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    mn_s, template_s, anchors_s = vec2parms(solution.x, p_len=100, am_count=am_count)
    modulation_s = create_modulation(
        anchors=anchors_s, samples=len(data), kind=am_interp
    )
    return mn_s, template_s, modulation_s


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
    mn_s, template_s, modulation_s = fit_with_plen_known(
        data, p_len=100, am_count=ac, am_interp="cubic"
    )
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(template, linewidth=5, color="gray")
    ax[0].plot(template_s, linewidth=1, color="red")
    ax[1].plot(modulation, linewidth=5, color="gray")
    ax[1].plot(modulation_s, linewidth=1, color="red")

    print(np.corrcoef(modulation_s, modulation)[0, 1])
    print(np.corrcoef(template_s, template)[0, 1])
