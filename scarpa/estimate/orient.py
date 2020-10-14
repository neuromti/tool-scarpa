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
    data: ndarray, max_iter: int = 10, demean: bool = True, scale: bool = True,
) -> ndarray:
    """orient data so that the first principal component exhibits only positive coefficients
    
    args
    ----
    data: ndarray
        the matrix of variables x observations
    max_iter:int
        how many iterations to run at most
    demean: bool
        whether to center the data
    scale: bool
        whether to divive the data by its std

    returns
    -------
    flipper: ndarray
        the array to flip the data matrix so that the first component has always positive coefficients
   
    """

    def newflip(data):
        return np.ones((data.shape[1],))

    data = data.copy()
    if demean:
        data -= data.mean(axis=0)
    if scale:
        data /= data.std(axis=0, ddof=1)

    # flip individual trials until the first component has only positive
    # first iteration
    c, s, v = pca_largest(data, demean=False, scale=False)
    flipper = newflip(data) * np.sign(c[:, 0])
    finalflipper = newflip(data)
    data *= flipper
    finalflipper *= flipper
    iterations = 0
    while not np.all(flipper > 0) and iterations < max_iter:
        c, s, v = pca_largest(data, demean=False, scale=False)
        flipper = newflip(data) * np.sign(c)
        data *= flipper
        finalflipper *= flipper
        iterations += 1

    return finalflipper
