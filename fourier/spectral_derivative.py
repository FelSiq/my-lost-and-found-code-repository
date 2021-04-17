import typing as t

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def spectral_derivative(X, X_lims=(0, 1)):
    n = len(X)
    z = np.fft.fft(X)
    L = X_lims[1] - X_lims[0]

    fundamental_freq = 2.0 * np.pi / L
    kappa = fundamental_freq * np.arange(-n / 2, n / 2)
    kappa = np.fft.fftshift(kappa)

    deriv = np.real(np.fft.ifft(1j * kappa * z))

    return deriv


def _test():
    import matplotlib.pyplot as plt

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.square(a[5:-5] - b[5:-5]))))

    f = lambda t: 0.6 * np.cos(2 * np.pi * 0.5 * t)
    df = lambda t: -0.6 * np.sin(2 * np.pi * 0.5 * t) * 2 * np.pi * 0.5
    g = lambda t: np.exp(-(t ** 2)/2 + 4)
    dg = lambda t: np.exp(-(t ** 2)/2 + 4) * -2 * t / 2
    h = lambda t: 0.4 * np.sin(2 * np.pi * 3 * t + 0.75)
    dh = lambda t: 0.4 * np.cos(2 * np.pi * 3 * t + 0.75) * 2 * np.pi * 3

    fX = lambda x: f(x) * g(x) + h(x)
    dfX = lambda x: f(x) * dg(x) + g(x) * df(x) + dh(x)

    t = np.linspace(-2, 2, 100)
    eps = 1e-7

    e_finite_diff = np.zeros_like(t)
    for i, x in enumerate(t):
        e_finite_diff[i] = (fX(x + eps) - fX(x)) / eps

    e2_finite_diff = np.zeros_like(t)
    for i, x in enumerate(t):
        e2_finite_diff[i] = 0.5 * (fX(x + eps) - fX(x - eps)) / eps

    X = fX(t) # + 0.5 * np.random.randn(t.size)
    dX = dfX(t)

    data_finite_diff = np.zeros_like(t)
    dt = t[1] - t[0]
    for i in np.arange(1, t.size - 1):
        data_finite_diff[i] = 0.5 * (X[i+1] - X[i-1]) / dt

    data_finite_diff[0] = data_finite_diff[1]
    data_finite_diff[-1] = data_finite_diff[-2]

    spectral = spectral_derivative(X, (-2, 2))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 10))

    ax1.set_title(f"Spectral derivative: {rmse(dX, spectral):.8f}")
    ax1.plot(dX, label="analytical")
    ax1.plot(spectral, label="spectral")
    ax1.set_xticks([])
    ax1.legend()

    ax2.set_title(f"2$\epsilon$ finite diff: {rmse(dX, e2_finite_diff):.8f}")
    ax2.plot(dX, label="analytical")
    ax2.plot(e2_finite_diff, label="2$\epsilon$ finite diff")
    ax2.set_xticks([])
    ax2.legend()

    ax3.set_title(f"$\epsilon$ finite diff: {rmse(dX, e_finite_diff):.8f}")
    ax3.plot(dX, label="analytical")
    ax3.plot(e_finite_diff, label="$\epsilon$ finite diff")
    ax3.set_xticks([])
    ax3.legend()

    ax4.set_title(f"2$\epsilon$ data finite diff: {rmse(dX, data_finite_diff):.8f}")
    ax4.plot(dX, label="analytical")
    ax4.plot(data_finite_diff, label="2$\epsilon$ data finite diff")
    ax4.set_xticks([])
    ax4.legend()

    plt.show()


if __name__ == "__main__":
    _test()
