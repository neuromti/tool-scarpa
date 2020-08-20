from numpy import ndarray
import numpy as np
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
from scipy.optimize import minimize
from pprint import pprint
from scipy.linalg import norm
from scipy.stats import special_ortho_group


def constrain_identity(x, k):

    R = x.reshape(k, k)
    eq = norm(R.dot(R.T) - eye(k, k))
    return eq


def rotate_varimax(Phi, gamma=1, max_iter=20, tol=1e-6):
    "https://en.wikipedia.org/wiki/Talk:Varimax_rotation"

    p, k = Phi.shape
    R = eye(k)
    d = 0
    i = 0
    for i in range(max_iter):
        d_old = d
        Lambda = dot(Phi, R)
        V = asarray(Lambda) ** 3 - (gamma / p) * dot(
            Lambda, diag(diag(dot(Lambda.T, Lambda)))
        )
        u, s, vh = svd(dot(Phi.T, V,))
        R = dot(u, vh)
        d = sum(s)
        if d - d_old < tol:
            break

    print("Varimax rotation finished after iteration", i)
    pprint(R)
    return R


def rotate_hilbert(_scores, max_iter=20, tol=1e-6):
    "rotate scores to maximize their phase-shift to 90°"

    def objective(x, _scores: ndarray, k: int):
        from scipy.signal import hilbert

        R = x.reshape(k, k)
        scores = R.dot(_scores.T)
        select = np.triu(np.ones((k + 1, k + 1)), 1).flatten() == 1

        best = []
        for six, score in enumerate(scores):
            deriv = np.imag(hilbert(score))
            rvals = np.abs(np.corrcoef(scores, deriv))
            rvals = [v for s, v in zip(select, rvals.flatten()) if s]
            _best = np.max(np.abs(rvals))
            best.append(_best)
        # the closer to one, the better, but we need to invert for minimize
        cost = len(best) - np.sum(best)
        return cost

    p, k = _scores.shape
    initial = special_ortho_group.rvs(k).flatten()
    cons = [{"type": "eq", "fun": constrain_identity, "args": [k]}]
    bnds = [(-1.01, 1.01)] * len(initial)
    solution = minimize(
        objective,
        args=(_scores, k),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )

    R = solution.x.reshape(k, k)
    print("Hilbert Rotation finished after iteration", solution.nit, "with")
    pprint(R)
    return R


def rotate_shape(_scores, shape: ndarray = None, max_iter=20, tol=1e-10):
    "rotate scores to maximize their phase-shift to 90°"

    def objective(x, _scores: ndarray, shape: ndarray, k: int):
        from scipy.signal import hilbert

        R = x.reshape(k, k)
        scores = R.dot(_scores.T)
        select = np.triu(np.ones((k + 1, k + 1)), 1).flatten() == 1

        rvals = np.abs(np.corrcoef(scores, shape))
        rvals = [v for s, v in zip(select, rvals.flatten()) if s]
        best = np.max(np.abs(rvals))
        # the closer to one, the better, but we need to invert for minimize
        cost = 1 - best
        # print("Shapecost: ", cost)
        return cost

    if shape is None:
        from scarpa.generate.shapes import sinus

        print("Defaulting to sinus")
        shape = sinus(len(_scores))
    p, k = _scores.shape
    initial = special_ortho_group.rvs(k).flatten()
    cons = [{"type": "eq", "fun": constrain_identity, "args": [k]}]
    bnds = [(-1.01, 1.01)] * len(initial)
    solution = minimize(
        objective,
        args=(_scores, shape, k),
        x0=initial,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )

    R = solution.x.reshape(k, k)
    print("Shape Rotation finished after iteration", solution.nit, "with")
    pprint(R)
    return R
