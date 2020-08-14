from numpy import ndarray


def factoran(data: ndarray, n_factors: int = 2, demean=True, scale=True):
    from sklearn.decomposition import FactorAnalysis

    data = data.copy()
    if demean:
        data -= data.mean(0)
    if scale:
        data /= np.std(data, 0)

    fa = FactorAnalysis(n_components=n_factors)
    scores = fa.fit_transform(data)
    coeffs = fa.components_.T
    return coeffs, scores


def varimax(
    data: ndarray, n_factors: int = 2, demean=True, scale=True, plot=False
):
    """
    """

    def rotate(Phi, gamma=1, max_iter=20, tol=1e-6):
        "https://en.wikipedia.org/wiki/Talk:Varimax_rotation"
        from numpy import eye, asarray, dot, sum, diag
        from numpy.linalg import svd

        p, k = Phi.shape
        R = eye(k)
        d = 0
        for i in range(max_iter):
            d_old = d
            Lambda = dot(Phi, R)
            u, s, vh = svd(
                dot(
                    Phi.T,
                    asarray(Lambda) ** 3
                    - (gamma / p)
                    * dot(Lambda, diag(diag(dot(Lambda.T, Lambda)))),
                )
            )
            R = dot(u, vh)
            d = sum(s)
            if d - d_old < tol:
                break

        print("Rotation finished after iteration", i)
        return dot(Phi, R), R

    _coeffs, _scores, = factoran(
        data, n_factors=n_factors, demean=demean, scale=scale
    )
    coeffs, R = rotate(_coeffs, gamma=1, max_iter=100)
    scores = R.dot(_scores.T).T
    # scores = (coeffs.T.dot(data.T)).T

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(_coeffs[:, 0], _coeffs[:, 1])
        ax.scatter(coeffs[:, 0], coeffs[:, 1])
    return coeffs, scores


def _factoran_sklearn(data: ndarray, n_factors: int = 2):
    from sklearn.decomposition import FactorAnalysis

    fa = FactorAnalysis(n_components=n_factors, svd_method="lapack")
    scores = fa.fit_transform(data)
    return scores


def _factoran_statsmodels(data: ndarray, n_factors: int = 2):
    from statsmodels.multivariate.factor import Factor

    fa = Factor(endog=data, n_factor=n_factors, method="pa")
    result = fa.fit()
    return result.factor_scoring()


def _factoran_fa(data: ndarray, n_factors: int = 2):
    from factor_analyzer import FactorAnalyzer

    fa = FactorAnalyzer(n_factors=n_factors, rotation=None)
    score = fa.fit_transform(data)
    return score


def _factoran(
    data: ndarray, n_components: int = 2, backend: str = "statsmodels"
):
    if backend == "sklearn":
        return _factoran_sklearn(data, n_components)
    elif backend == "statsmodels":
        return _factoran_statsmodels(data, n_components)
    elif backend == "fa":
        return _factoran_fa(data, n_components)
    else:
        raise ValueError(f"Backend {backend} is unknown")


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scarpa.estimate.pca import pca

    # a frequency drift of only .1% is enough
    data = np.atleast_2d(
        np.sin(1.001 * 2 * np.pi * np.linspace(0, 20, 2000))
    ).T
    # but at a drift of 1% makes it clear
    data = np.atleast_2d(np.sin(1.01 * 2 * np.pi * np.linspace(0, 20, 2000))).T
    data += np.random.normal(0, 0.1, (data.shape))
    data = np.reshape(data, (20, 100)).T

    # plt.plot(data)
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(data.mean(1))
    ax[1].plot(factoran(data, 2)[1])
    ax[2].plot(varimax(data, 2)[1])

    c, s, v = pca(data)
    ax[3].plot(s[:, 0:3])
