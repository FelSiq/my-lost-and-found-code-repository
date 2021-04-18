"""TODO: clean this code"""
import numpy as np


def polymult(coeffs_f, coeffs_g):
    n = len(coeffs_f) + len(coeffs_g) - 1
    c_f_transf = np.fft.fft(coeffs_f, n=n)
    c_g_transf = np.fft.fft(coeffs_g, n=n)
    c_h = np.real(np.fft.ifft(c_g_transf * c_f_transf))
    return c_h


def _test():
    import matplotlib.pyplot as plt

    coeffs_f = np.random.randint(-4, 4, size=np.random.randint(1, 6))
    coeffs_g = np.random.randint(-4, 4, size=np.random.randint(1, 6))

    f = lambda x: np.polyval(coeffs_f, x)
    g = lambda x: np.polyval(coeffs_g, x)

    t = np.linspace(-1, 1, 100)
    comb = f(t) * g(t)

    coeffs_h = polymult(coeffs_g, coeffs_f)

    h = lambda x: np.polyval(coeffs_h, x)

    ht = h(t)
    plt.title(f"rmse: {np.sqrt(np.mean(np.square(ht - comb))):.4f}")
    plt.plot(t, ht, label="conv")
    plt.plot(t, comb, label="ref")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test()
