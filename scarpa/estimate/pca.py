from numpy.core.fromnumeric import argsort
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import ndarray

#%%
def pca(data: ndarray, rotation: str = "positive"):
    """perform principal component analysis on the raw data matrix
    
    args
    ----
    data: ndarray
        the matrix of variables x observations
    
    returns
    -------
    eigen_vector:ndarray
        the eigenvectors or coefficients of each component
    score: ndarray
        the score of each component
    eigen_value:
        the eigenvalues of each component
    """
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)

    eigen_value, eigen_vector = eigh(R)

    idx = np.argsort(eigen_value)[::-1]
    eigen_vector = eigen_vector[:, idx]
    eigen_value = eigen_value[idx]

    eigen_value /= eigen_value.sum(axis=0)
    score = np.dot(eigen_vector.T, data.T).T

    if rotation == "positive":
        for ei, ev in enumerate(eigen_vector):
            if ev.mean() < 0:
                eigen_vector[:, ei] *= -1
                score[:, ei] *= -1
    return eigen_vector, score, eigen_value


def pca_largest(data: ndarray):
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    eigen_value, eigen_vector = eigsh(R, 1, which="LM")
    score = np.dot(eigen_vector.T, data.T).T
    # return eigen_vector[:,0], score[:,0], eigen_value
    return eigen_vector, score, eigen_value


def pca_reduce(data: ndarray, dimensions=1):
    x = np.asarray(data)

    if dimensions == 1:
        c, s, l = pca_largest(x.T)
        c = c[:, 0]
        s = s[:, 0]
        if c[1] < 0:
            c = -c
            s = -s
        return s, c
    else:

        pca = decomposition.PCA()
        x_std = StandardScaler().fit_transform(x)
        pca.fit_transform(x_std)
        coeffs = pca.components_[:, :dimensions].copy()
        scores = pca.fit_transform(x.T)
        return scores[:, :dimensions], coeffs


def remove_pca(data: ndarray, periodlen: int, fs: int):
    pass


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.atleast_2d(np.sin(2 * np.pi * np.linspace(0.5, 20.5, 2000))).T
    data += np.random.normal(0, 0.01, (data.shape))
    data = np.reshape(data, (20, 100)).T

    # plt.plot(data)
    fig, ax = plt.subplots(5, 1)
    ax[0].plot(data.mean(1))
    c, s, v = pca(data)
    ax[1].plot(s[:, 0:2])
