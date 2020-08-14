"generate a signal vector using templates"
import numpy as np
from numpy import ndarray


def stack_template(template: ndarray, periodcount: float) -> ndarray:
    """stack a template periodically
    
    args
    ----

    template: ndarray
        the waveform along one periodic

    periodcount: float
        the duration of the complete signal in periods. Fractions are allowed
    
    returns
    -------
    model: ndarray
        the generated vector

    """
    if periodcount <= 0:
        raise ValueError("Periodcount can not be smaller than 0")

    plen = len(template)
    reps, remainder = int(periodcount // 1), int(plen * (periodcount % 1))
    model = np.empty(0)
    for rep in range(reps):
        model = np.hstack((model, template))

    model = np.hstack((model, template[:remainder]))

    return model


def activate_template(template: ndarray, actvect: ndarray) -> ndarray:
    """convolve an activation vector with a period template
    
    args
    ----

    template: ndarray
        the waveform

    actvect: float
        the activation vector

    returns
    -------
    model: ndarray
        the generated vector

    """

    return np.convolve(actvect, template, mode="same")


if __name__ == "__main__":
    # benchmark
    from timeit import repeat

    template = np.ones(100)
    periodcount = 100
    actvect = np.zeros(len(template) * periodcount)
    for i in range(0, len(actvect), len(template)):
        actvect[i] = 1

    def bench():
        stack_template(template, periodcount)

    setup = "from __main__ import bench"

    reps = 100
    number = 100
    bench_out = repeat(stmt="bench()", setup=setup, number=number, repeat=reps)
    bench_out = [b * 1000 / number for b in bench_out]
    print(
        "stack_template: {0:3.2f} ({1:3.2f}) ms".format(
            np.mean(bench_out), np.std(bench_out)
        )
    )

    def bench():
        activate_template(template, actvect)

    bench_out = repeat(stmt="bench()", setup=setup, number=number, repeat=reps)
    bench_out = [b * 1000 / number for b in bench_out]
    print(
        "activate_template: {0:3.2f} ({1:3.2f}) ms".format(
            np.mean(bench_out), np.std(bench_out)
        )
    )
