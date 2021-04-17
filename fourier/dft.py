import numpy as np
import sklearn.base


class DFT(sklearn.base.TransformerMixin):
    def __init__(self):
        self.dft_mat = np.empty(0)
        self.n = -1
        self.fundamental_freq_n = 0.0

    def fit(self, X, y=None):
        self.n = len(X)
        self.fundamental_freq_n = np.exp(-2.0j * np.pi / self.n)
        self.dft_mat = (
            np.vander(
                np.power(self.fundamental_freq_n, np.arange(self.n)),
                increasing=True,
            )
            / np.sqrt(self.n)
        )

        return self

    def transform(self, X, y=None):
        X = np.asarray(X).ravel()
        X = np.expand_dims(X, axis=1)
        coeffs = np.dot(self.dft_mat, X)
        return np.squeeze(coeffs)

    def inv_transform(self, X, y=None):
        X = np.asarray(X).ravel()
        X = np.expand_dims(X, axis=1)
        fX = np.dot(np.conjugate(self.dft_mat).T, X)
        return np.squeeze(fX)


def _test():
    import matplotlib.pyplot as plt

    f = lambda t: 0.6 * np.cos(2 * np.pi * 0.5 * t)
    g = lambda t: np.exp(-(t ** 2) / 2 + 4)
    h = lambda t: 0.4 * np.sin(2 * np.pi * 3 * t + 0.75)
    fX = lambda x: f(x) * g(x) + h(x)

    t = np.linspace(-5, 5, 1000)
    X = fX(t)

    model = DFT()
    coeffs = model.fit_transform(X)
    coeffs_aux = np.fft.fft(X, norm="ortho")

    X_aux = model.inv_transform(coeffs)

    plt.plot(X, label="original")
    plt.plot(np.real(X_aux), label="reconstructed")
    plt.legend()

    assert np.allclose(coeffs, coeffs_aux)
    assert np.allclose(X, X_aux)

    plt.show()


if __name__ == "__main__":
    _test()
