from scarpa.estimate.pca import pca_largest
from scipy.linalg import norm
from numpy import ndarray
import numpy as np


def orient_cs(coeffs: ndarray, scores: ndarray):
    """orient all components so that their average coefficient is positive

    args
    ----
    coeffs:ndarray
        the eigenvectors or coefficients of each component
    score: ndarray
        the score of each component
    
    returns
    -------
    coeffs:ndarray
        the oriented eigenvectors or coefficients of each component
    score: ndarray
        the oriented score of each component
    """

    coeffs = coeffs.copy()
    scores = scores.copy()
    for idx, c in enumerate(coeffs.T):
        if c.mean() < 0:
            coeffs[:, idx] *= -1
            scores[:, idx] *= -1
    return coeffs, scores


def orient_principal(
    data: ndarray,
    col: float = 1e-6,
    max_iter: int = 10,
    demean: bool = True,
    scale: bool = True,
):
    """orient data so that the first principal component exhibits only positive coefficients
    
    args
    ----
    data: ndarray
        the matrix of variables x observations
    col:float
        the convergence criterion
    max_iter:int
        how many iterations to run at most
    demean: bool
        whether to center the data
    scale: bool
        whether to divive the data by its std

    returns
    -------
    data: ndarray
        the oriented matrix of variables x observations
   
    """

    data = data.copy()
    if demean:
        data -= data.mean(axis=0)
    if scale:
        data /= data.std(axis=0, ddof=1)

    c, s, v = pca_largest(data, demean=False, scale=False)
    if c.mean() < 0:
        data = -data
        c, s, v = pca_largest(data, demean=False, scale=False)
    iterations = 0
    convergence = 2
    flipper = np.ones(data.shape[1])
    # flip individual trials until the first component has only positive
    # coefficients
    while convergence > col:
        iterations += 1
        old_c = c
        flipper = np.ones(data.shape[1])
        for tix, t in enumerate(old_c):
            if t < 0:
                flipper[tix] = -1

        data = flipper * data
        c, s, v = pca_largest(data, demean=False, scale=False)
        cm = c.mean()
        if cm < 0:
            data = -data
            c, s, v = pca_largest(data, demean=False, scale=False)
            cm = c.mean()
        convergence = norm(old_c - c)
        if iterations > max_iter:
            break

    # make sure the negative peak comes first
    if s.argmax() < s.argmin():
        data *= -1
    return data
