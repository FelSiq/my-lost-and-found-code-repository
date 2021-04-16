import numpy as np
import scipy.integrate

import sys

sys.path.append("../geometry")
import function_projection


class _FourierApproxBase:
    def __init__(self, max_components: int = 200, L: float = 1.0):
        assert int(max_components) > 0
        assert L > 0.0

        self.max_components = int(max_components)
        self.L = L

        self.A0, self.Ak, self.Bk = np.empty((3, 0))

    def transform(self, X):
        raise NotImplementedError


class FastFourierApprox(_FourierApproxBase):
    def transform(self, X):
        X = np.asarray(X)
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        k = np.arange(1, 1 + self.max_components)
        combs = 2.0 * np.pi / self.L * np.outer(k, t)
        cos = np.cos(combs)
        sin = np.sin(combs)

        # Note: norm_sqr_cos_sin = 1 / function_projection.func_norm(cos, dt)
        #                        = 1 / function_projection.func_norm(sin, dt)
        norm_sqr_cos_sin = 2.0 / self.L

        self.A0 = np.squeeze(scipy.integrate.trapezoid(X, dx=dt) * norm_sqr_cos_sin)

        self.Ak = (
            scipy.integrate.trapezoid(X * cos, dx=dt, axis=1)[:, np.newaxis]
            * norm_sqr_cos_sin
        )

        self.Bk = (
            scipy.integrate.trapezoid(X * sin, dx=dt, axis=1)[:, np.newaxis]
            * norm_sqr_cos_sin
        )

        data_max_comp = (X.size - 2) // 2
        self.Ak[data_max_comp:] = self.Bk[data_max_comp:] = 0.0

        ff = 0.5 * self.A0 + np.sum(self.Ak * cos + self.Bk * sin, axis=0)

        self.Ak = self.Ak.ravel()
        self.Bk = self.Bk.ravel()

        return ff


class SlowFourierApprox(_FourierApproxBase):
    def _compute_Ak_coeff(self, X: np.ndarray, t: np.ndarray, dt: float, k: int):
        cos = np.cos(2.0 * np.pi / self.L * k * t)
        proj = function_projection.project_f_onto_g(f=X, g=cos, dx=dt)
        return proj, cos

    def _compute_Bk_coeff(self, X: np.ndarray, t: np.ndarray, dt: float, k: int):
        sin = np.sin(2.0 * np.pi / self.L * k * t)
        proj = function_projection.project_f_onto_g(f=X, g=sin, dx=dt)
        return proj, sin

    def transform(self, X):
        X = np.asarray(X)
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        ff, _ = self._compute_Ak_coeff(X, t, dt, k=0)
        self.A0 = 2.0 * ff[0]
        self.Ak, self.Bk = np.zeros((2, self.max_components))

        for k in np.arange(1, 1 + min((X.size - 2) // 2, self.max_components)):
            proj_cos, cos = self._compute_Ak_coeff(X, t, dt, k)
            proj_sin, sin = self._compute_Bk_coeff(X, t, dt, k)

            # Note: could be any index since Ak and Bk are all the same for all t
            Ak = proj_cos[-1] / (1e-7 + cos[-1])
            Bk = proj_sin[-1] / (1e-7 + sin[-1])

            ff += proj_cos + proj_sin

            self.Ak[k - 1] = Ak
            self.Bk[k - 1] = Bk

        return ff


def _test():
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 200 + int(np.random.random() > 0.5))
    trend = np.linspace(0, 5 * np.random.random(), x.size // 2)
    X = (
        0.3 * np.cos(1.5 * x + 0.5)
        + 0.7 * np.sin(3 * x)
        + 0.2 * np.sin(2 * np.pi / 0.5 * x)
    )
    # X += 0.1 * np.random.randn(x.size)
    X[: x.size // 2] += trend
    X[(x.size + 1) // 2 :] += trend[-1] - trend

    ref = FastFourierApprox(L=2 * np.pi)
    ref_slow = SlowFourierApprox(L=2 * np.pi)

    approx = ref.transform(X)
    approx_slow = ref_slow.transform(X)

    diff = np.abs(approx[5:-5] - X[5:-5])
    aux = diff.argmax()
    print(X.size, aux, diff[aux])
    assert np.allclose(approx, approx_slow)
    print(f"RMSE: {np.sqrt(np.mean(np.square(approx[5:-5] - X[5:-5]))):.4f}")

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.set_title("Fast Approximation")
    ax1.plot(X, label="original")
    ax1.plot(approx, label="approx fast")
    ax1.legend()

    ax2.set_title("Slow Approximation")
    ax2.plot(X, label="original")
    ax2.plot(approx_slow, label="approx slow")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    _test()
