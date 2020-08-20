from numpy import ndarray
import scipy.signal as sg
import numpy as np


def validate(coeffs, scores, data):
    return ((coeffs.dot(scores.T).ravel() - data.T.ravel()) ** 2).mean()


def factoran(data: ndarray, n_factors: int = 2, demean=True, scale=True):
    from sklearn.decomposition import FactorAnalysis

    data = data.copy()
    if demean:
        data -= data.mean(axis=0)
    if scale:
        data /= data.std(axis=0, ddof=1)

    fa = FactorAnalysis(n_components=n_factors)
    # validation
    # from sklearn.model_selection import cross_val_score
    # cross_val_score(fa, data).mean()
    scores = fa.fit_transform(data)
    coeffs = np.sqrt(0.5) * fa.components_.T
    return coeffs, scores


def derimax(
    data: ndarray,
    n_factors: int = 2,
    demean=True,
    scale=True,
    max_iter: int = 200,
    plot=False,
):
    """
    """
    from scarpa.estimate.rotation import rotate_hilbert

    _coeffs, _scores, = factoran(
        data, n_factors=n_factors, demean=demean, scale=scale
    )
    R = rotate_hilbert(_scores, max_iter=max_iter)
    coeffs = np.dot(_coeffs, R)
    scores = R.dot(_scores.T).T

    if plot:
        fig, ax = plt.subplots(n_factors, n_factors)
        for a in range(n_factors):
            for b in range(n_factors):
                ax[a, b].scatter(_coeffs[:, a], _coeffs[:, b], color="k")
                ax[a, b].scatter(coeffs[:, a], coeffs[:, b], color="r")
    return coeffs, scores


def shapemax(
    data: ndarray,
    n_factors: int = 2,
    shape: ndarray = None,
    demean=True,
    scale=True,
    max_iter: int = 200,
    plot=False,
):
    """
    """
    from scarpa.estimate.rotation import rotate_shape
    from scarpa.generate.shapes import sinus

    _coeffs, _scores, = factoran(
        data, n_factors=n_factors, demean=demean, scale=scale
    )
    R = rotate_shape(_scores, shape=shape, max_iter=max_iter)
    coeffs = np.dot(_coeffs, R)
    scores = R.dot(_scores.T).T

    if plot:
        fig, ax = plt.subplots(n_factors, n_factors)
        for a in range(n_factors):
            for b in range(n_factors):
                ax[a, b].scatter(_coeffs[:, a], _coeffs[:, b], color="k")
                ax[a, b].scatter(coeffs[:, a], coeffs[:, b], color="r")
    return coeffs, scores


def varimax(
    data: ndarray,
    n_factors: int = 2,
    demean=True,
    scale=True,
    max_iter: int = 200,
    plot=False,
):
    """Perform factor analysis with varimax rotation
    
    """

    from scarpa.estimate.rotation import rotate_varimax

    _coeffs, _scores, = factoran(
        data, n_factors=n_factors, demean=demean, scale=scale
    )
    R = rotate_varimax(_coeffs, gamma=1, max_iter=max_iter)
    coeffs = np.dot(_coeffs, R)
    scores = R.dot(_scores.T).T
    # scores = (coeffs.T.dot(data.T)).T

    if plot:
        fig, ax = plt.subplots(n_factors, n_factors)
        for a in range(n_factors):
            for b in range(n_factors):
                ax[a, b].scatter(_coeffs[:, a], _coeffs[:, b], color="k")
                ax[a, b].scatter(coeffs[:, a], coeffs[:, b], color="r")
    return coeffs, scores


if __name__ == "__main__":

    def derive(data: ndarray) -> ndarray:
        from scipy.signal import hilbert

        out = data.copy().T
        for cix, clm in enumerate(out):
            out[cix, :] = np.imag(hilbert(clm))
        return out.T

    import numpy as np
    import matplotlib.pyplot as plt
    from scarpa.estimate.pca import pca
    from scarpa.generate.shapes import hanning, sawtooth, mexicanhat
    import scipy.signal as sg

    n_level = 0.0
    m_level = 0.1
    p_count = 50
    p_len = 100
    s_len = p_count * p_len
    sig = np.atleast_2d(np.sin(2 * np.pi * np.linspace(0, p_count, s_len))).T
    sig += np.random.normal(0, n_level, (sig.shape))
    data = np.reshape(sig, (p_count, p_len)).T

    # plt.plot(data)
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].plot(sig)
    ax[0, 1].plot(data.mean(1))
    c, scores = factoran(data, 2)
    ax[1, 0].plot(c)
    ax[1, 1].plot(scores)
    c, scores = varimax(data, 2)
    ax[2, 0].plot(c)
    ax[2, 1].plot(scores)
    c, scores = derimax(data, 2)
    ax[3, 0].plot(c)
    ax[3, 1].plot(scores)

    fig, ax = plt.subplots(1, 1)
    c, scores = factoran(data, 2)
    fit = c.dot(scores.T).ravel()
    ndata = data.copy()
    ndata - +data.mean(0)
    ndata / +data.std(0)
    ax.plot(ndata.T.ravel())
    ax.plot(ndata.T.ravel() - fit)

    print("Perfect sinus")
    for i in range(1, 10, 1):
        print(i, f"{validate(*factoran(data, i), data):3.4f}")

    # if frequency is not fitting into the window, i.e. period is not an
    # integer multiple, it will leak into two sinusoids. THis makes sense, as
    # otherwise a single sinusoid could easily catch the stationary perfectly
    # periodic signal
    # a frequency drift of only .1% is enough to induce this decomposition into
    # two basis vectors. Note also that they are (obviously orthognal), but
    # also almost perfectly 90° off as calculate by taking the hilbert
    # transformation

    # a drift of 1% makes it very clear
    sig = np.atleast_2d(
        np.sin(2 * np.pi * np.linspace(0, p_count * 1.01, s_len))
    ).T
    sig += np.random.normal(0, n_level, (sig.shape))
    data = np.reshape(sig, (p_count, p_len)).T

    # plt.plot(data)

    fig, ax = plt.subplots(4, 2)
    ax[0, 0].plot(sig)
    ax[0, 1].plot(data.mean(1))
    c, scores = factoran(data, 4)
    derivatives = derive(scores)
    ax[1, 0].plot(c)
    ax[1, 1].plot(derivatives, linewidth=4, color="gray")
    ax[1, 1].plot(scores)
    c, scores = varimax(data, 4)
    derivatives = derive(scores)
    ax[2, 0].plot(c)
    ax[2, 1].plot(derivatives, linewidth=4, color="gray")
    ax[2, 1].plot(scores)
    c, scores = derimax(data, 4)
    derivatives = derive(scores)
    ax[3, 0].plot(c)
    ax[3, 1].plot(derivatives, linewidth=4, color="gray")
    ax[3, 1].plot(scores)

    fig, ax = plt.subplots(1, 1)
    c, scores = factoran(data, 2)
    fit = c.dot(scores.T).ravel()
    ndata = data.copy()
    ndata - +data.mean(0)
    ndata / +data.std(0)
    ax.plot(ndata.T.ravel())
    ax.plot(ndata.T.ravel() - fit)

    print("Frequency drift")
    for i in range(1, 10, 1):
        print(i, f"{validate(*factoran(data, i), data):3.4f}")

    # similarily, amplitude modulation
    sig = np.atleast_2d(
        np.sin(2 * np.pi * np.linspace(0, p_count * 1.0, s_len))
    ).T

    sig += np.random.normal(0, n_level, (sig.shape))
    sig += (
        m_level
        * np.atleast_2d(
            np.sin(2 * np.pi * np.linspace(0, p_count * 3, s_len))
        ).T
    )
    sig = np.atleast_2d(sig[:, 0] * (1 + hanning(sig.shape[0]))).T
    # sig[800:850] += hanning(50)[:, np.newaxis]
    data = np.reshape(sig, (p_count, p_len)).T

    # plt.plot(data)
    # can't be perfect derivatives because different frequencies!
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].plot(sig)
    ax[0, 1].plot(data.mean(1))
    c, scores = factoran(data, 2)
    ax[1, 0].plot(c)
    ax[1, 1].plot(scores)
    c, scores = shapemax(data, 2)
    ax[2, 0].plot(c)
    ax[2, 1].plot(scores)
    c, scores = derimax(data, 2)
    ax[3, 0].plot(c)
    ax[3, 1].plot(scores)

    fig, ax = plt.subplots(1, 1)
    c, scores = factoran(data, 2)
    fit = c.dot(scores.T).ravel()
    ndata = data.copy()
    ndata - +data.mean(0)
    ndata / +data.std(0)
    ax.plot(ndata.T.ravel())
    ax.plot(ndata.T.ravel() - fit)

    print("Amplitude modulation")
    for i in range(1, 10, 1):
        print(i, f"{validate(*factoran(data, i), data):3.4f}")

    # mixture
    sigA = np.atleast_2d(
        np.sin(2 * np.pi * np.linspace(0, p_count * 1, s_len))
    ).T
    sigA = np.atleast_2d(sigA[:, 0] * (1 + sawtooth(sig.shape[0]))).T

    sigB = np.atleast_2d(
        sg.sawtooth(2 * np.pi * np.linspace(0, p_count * 3, s_len))
    ).T
    sigB = np.atleast_2d(sigB[:, 0] * (1 - sawtooth(sig.shape[0]))).T

    sig = sigA + sigB
    sig += np.random.normal(0, n_level, (sig.shape))

    # sig[800:850] += hanning(50)[:, np.newaxis]
    data = np.reshape(sig, (p_count, p_len)).T

    # plt.plot(data)
    # can't be perfect derivatives because different frequencies!
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].plot(sig)
    ax[0, 1].plot(data.mean(1))
    c, scores = factoran(data, 2)
    ax[1, 0].plot(c)
    ax[1, 1].plot(scores)
    c, scores = shapemax(data, 2)
    ax[2, 0].plot(c)
    ax[2, 1].plot(scores)
    c, scores = derimax(data, 3)
    ax[3, 0].plot(c)
    ax[3, 1].plot(scores)

    c, scores = shapemax(data, 2)
    fig, ax = plt.subplots(scores.shape[1])
    for s, ax in zip(scores.T, ax.flatten()):
        ax.plot(s)

    fig, ax = plt.subplots(1, 1)
    fit = c.dot(scores.T).ravel()
    ndata = data.copy()
    ndata -= data.mean(0)
    ndata /= data.std(0)
    ax.plot(ndata.T.ravel())
    ax.plot(ndata.T.ravel() - fit)

    print("Mixture")
    for i in range(1, 10, 1):
        print(i, f"{validate(*factoran(data, i), data):3.4f}")
    # # %%
    # fig, ax = plt.subplots(3, 3)
    # for n_comp, ax in enumerate(ax.ravel(), 1):
    #     coeffs, scores = derimax(data, n_comp)
    #     ssq = sum((coeffs.dot(scores.T).ravel() - data.T.ravel()) ** 2)
    #     ax.plot(coeffs.dot(scores.T).ravel() - data.T.ravel())
    #     print(n_comp, ssq)

    # %%
    sig = np.sin(2 * np.pi * np.linspace(0, p_count, s_len))
    sig += np.random.normal(0, n_level, (sig.shape))
    sig *= 1 + hanning(sig.shape[0])

    w_len = 10 * p_len

    values = [0] * (9 * p_len)
    for i in range(w_len, len(sig), p_len):
        segment = sig[i - w_len : i].reshape(p_len, -1)
        c, s = factoran(segment, 3)
        recon = s.dot(c.T).ravel()
        val = (segment.ravel() - recon)[-p_len:].tolist()
        values.extend(val)

    plt.plot(sig), plt.plot(values)
