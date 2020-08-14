from scarpa.generate.template import stack_template
from scarpa.generate.modulation import create_modulation
from scarpa.estimate.contraints import (
    constraint_template_mean,
    constraint_template_range,
)
from scipy.signal import hilbert, correlate, medfilt
from scipy.linalg import norm
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numpy import ndarray
from typing import Tuple


class TemplateEstimator:
    from scarpa.generate.modulation import create_modulation
    from scarpa.generate.template import stack_template

    def __init__(
        self,
        period_len: int,
        n_anchorpoints: int,
        am_interp: str = "cubic",
        n_components: int = 1,
    ):
        self.p_len = period_len
        self.am_count = n_anchorpoints
        self.am_interp = am_interp
        self.n_components = n_components

    def fit(self, data):
        data = data.copy()
        N: int = len(data)
        self.p_count: float = N / self.p_len
        self.mn, templates, self.solution = fit(
            data,
            p_len=self.p_len,
            am_count=self.am_count,
            am_interp=self.am_interp,
            n_components=self.n_components,
        )

        return templates

    def evaluate(self):
        templates, anchors = recover_parms(
            self.solution, self.n_components, self.p_len, self.am_count
        )
        fit = evaluate(templates, self.p_count, anchors, self.am_interp)
        return fit + self.mn


def recover_parms(x, n_components, p_len, am_count) -> Tuple[ndarray, ndarray]:
    templates = x[0 : n_components * p_len]
    templates = templates.reshape(n_components, p_len)
    anchors = x[-am_count * n_components :]
    anchors = anchors.reshape(n_components, am_count)
    return templates, anchors


def evaluate(templates, p_count, anchors, am_interp):
    N = int(p_count * len(templates[0]))

    model = []
    for anchor, template in zip(anchors, templates):
        stacked = stack_template(template, p_count)
        modulation = create_modulation(anchor, samples=N, kind=am_interp)
        comp = stacked * modulation
        model.append(comp)
    fit = np.atleast_2d(model).sum(0)
    return fit


def objective(
    x: ndarray,
    data: ndarray,
    n_components: int,
    p_len: int,
    am_count: int,
    am_interp: str,
) -> float:
    N: int = len(data)
    p_count: float = N / p_len

    templates, anchors = recover_parms(x, n_components, p_len, am_count)
    fit = evaluate(templates, p_count, anchors, am_interp)
    residual = data - fit
    cost = norm(residual, 2)
    print(cost)
    return cost


def fit(
    data,
    p_len: int,
    am_count: int,
    am_interp: str = "nearest",
    n_components: int = 1,
) -> Tuple[float, ndarray, ndarray, ndarray]:
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
    n_components:int
        how many periodic templates should be fit to the data

    returns
    -------
    mn:float
        the mean of the data
    template: ndarray
        the estimated template waveform of the periodic signal
    modulation: ndarray
        the estimated amplitude modulation across the whole signal length

    """
    N: int = len(data)
    p_count: float = N / p_len

    # estimate the initial template values
    from scarpa.estimate.fa import factoran
    from scarpa.estimate.pca import pca_reduce

    periodic_data = (
        data[: p_len * int(p_count)].reshape((int(p_count), p_len)).T
    )
    # scores = factoran(periodic_data, n_components=n_components)
    scores, _ = pca_reduce(periodic_data.T, dimensions=n_components)
    templates = []
    # normalize scores
    for six, score in enumerate(scores.T):
        scores[:, six] = score / max(abs(score))

    # stack continuously into a vector for minimize
    templates = np.hstack(np.asanyarray(scores).T)
    t_len = n_components * p_len
    # estimate the initial amplitude modulation values
    am = abs(hilbert(data))
    am = [
        am[int(idx)]
        for idx in np.linspace(0, len(data), am_count, endpoint=False)
    ]
    am = am * n_components
    a_len = am_count * n_components
    # create the vector of initial values for minimize (argument there is x0)
    # count desc
    # 1     mean of the data vector
    # a     anchor_values
    # t     template
    initial = np.ones(t_len + a_len)
    initial[0:t_len] = templates
    initial[-a_len:] = am

    # initialize bounds
    # the template is bound between -1 and 1, and any amplitude is caused by
    # the am vector
    tb = (-1, 1)
    ab = (0, np.inf)
    bnds = [tb] * t_len + [ab] * (n_components * am_count)

    cons = [
        {"type": "eq", "fun": constraint_template_mean, "args": [p_len]},
        {"type": "eq", "fun": constraint_template_range, "args": [p_len]},
    ]

    mn_s = data.mean()
    data -= mn_s
    solution = minimize(
        objective,
        args=(data, n_components, p_len, am_count, am_interp),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    template_s, anchors_s = recover_parms(
        solution.x, n_components=n_components, p_len=p_len, am_count=am_count
    )
    print("Completed")
    return mn_s, template_s, solution.x


# %%
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
    modulation = create_modulation(
        anchors=anchors, samples=samples, kind="cubic"
    )
    data = pure * modulation

    p_len = 100
    am_count = 11
    am_interp = "cubic"
    te = TemplateEstimator(
        n_components=3, period_len=100, n_anchorpoints=11, am_interp="cubic"
    )
    template_s = te.fit(data)
    fit = te.evaluate()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(template, linewidth=5, color="gray")
    ax[0].plot(template_s.T, linewidth=1, color="red")

    ax[1].plot(data, linewidth=5, color="gray")
    ax[1].plot(fit, linewidth=1, color="red")

    print(np.corrcoef(data, fit)[0, 1])
