import numpy as np
from scipy import signal
from numpy import ndarray
from scipy.linalg import norm


def generate_sinus(frequency, duration, fs):
    t = np.linspace(0, duration, int(duration * fs))
    y = np.sin(2 * np.pi * t * frequency)
    return y


def generate_saw(frequency, duration, fs):
    t = np.linspace(0, duration, int(duration * fs))
    y = signal.sawtooth(2 * np.pi * t * frequency)
    return y


def calculate_kernel_duration(period: float, accuracy: int = 1) -> int:
    duration = period
    while np.round(duration, accuracy) != int(duration):
        duration += period
    return duration


def saw_kernel(frequency, fs):
    period = fs / frequency
    duration = calculate_kernel_duration(period)
    t = np.linspace(0, duration / fs, int(duration) + 1)
    y = signal.sawtooth(2 * np.pi * t * frequency)
    return y[:-1]


def sin_kernel(frequency, fs):
    period = fs / frequency
    duration = calculate_kernel_duration(period)
    t = np.linspace(0, duration / fs, int(duration) + 1)
    y = np.sin(2 * np.pi * t * frequency)
    return y[:-1]


def generate_signal(kernel: ndarray, sample_count: int):
    y = np.array(())
    while len(y) < sample_count:
        y = np.hstack((y, kernel))

    return y[0:sample_count]


def fit_signal(kernel: ndarray, sample_count: int, modulation: ndarray):
    y = generate_signal(kernel, sample_count)
    return y * modulation


def loss(y, y_hat) -> float:
    return norm(y - y_hat)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #
    # y = generate_sinus(1, 10, 1, 100)
    # mod = signal.hanning(y.shape[0])
    # plt.plot(y)
    # plt.plot(y * mod)
    #

    fs = 1000
    freq = 10
    duration = 1
    sample_count = int(duration * fs)

    k = sin_kernel(freq, fs)
    y = generate_signal(k, sample_count)
    mod = signal.hanning(sample_count)
    Y = y * mod
    modest = np.abs(signal.hilbert(Y))

    fig, ax = plt.subplots(1, 1)
    ax.plot(k)

    fig, ax = plt.subplots(1, 1)
    ax.plot(y)
    ax.plot(Y)
    ax.plot(modest)

    y_hat = fit_signal(kernel=k, sample_count=sample_count, modulation=modest)
    print(loss(y, y_hat))
    ax.plot(y_hat)
