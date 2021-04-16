import numpy as np
import scipy.integrate

import sys

sys.path.append("../geometry")
import function_projection


class _FourierApproxABBase:
    def __init__(self, max_components: int = 256, L: float = 1.0):
        assert int(max_components) > 0
        assert L > 0.0

        self.max_components = int(max_components)
        self.L = L

        self.A0, self.Ak, self.Bk = np.empty((3, 0))

    def transform(self, X):
        raise NotImplementedError


class _FourierApproxCBase:
    def __init__(self, max_components: int = 256, L: float = 1.0):
        assert int(max_components) > 0
        assert L > 0.0

        self.max_components = int(max_components)
        self.L = L

        self.C = np.empty(0)

    def transform(self, X):
        raise NotImplementedError


class FastFourierApproxAB(_FourierApproxABBase):
    def transform(self, X):
        X = np.asarray(X).ravel()
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        k = np.arange(1, 1 + self.max_components)
        combs = 2.0 * np.pi / self.L * np.outer(k, t)
        cos = np.cos(combs)
        sin = np.sin(combs)

        # Note: norm_sqr_cos_sin = 1 / function_projection.func_norm(cos, dt)
        #                        = 1 / function_projection.func_norm(sin, dt)
        norm_sqr_cos_sin = 2.0 / self.L

        X_norm = X * norm_sqr_cos_sin

        self.A0 = np.squeeze(scipy.integrate.trapezoid(X_norm, dx=dt))
        self.Ak = scipy.integrate.trapezoid(X_norm * np.conjugate(cos), dx=dt, axis=1)
        self.Bk = scipy.integrate.trapezoid(X_norm * np.conjugate(sin), dx=dt, axis=1)

        data_max_comp = (X.size - 2) // 2
        self.Ak[data_max_comp:] = self.Bk[data_max_comp:] = 0.0

        Ak = np.expand_dims(self.Ak, axis=1)
        Bk = np.expand_dims(self.Bk, axis=1)

        X_approx = 0.5 * self.A0 + np.sum(Ak * cos + Bk * sin, axis=0)

        return X_approx


class SlowFourierApproxAB(_FourierApproxABBase):
    def _compute_Ak_coeff(self, X: np.ndarray, t: np.ndarray, dt: float, k: int):
        cos = np.cos(2.0 * np.pi / self.L * k * t)
        proj = function_projection.project_f_onto_g(f=X, g=cos, dx=dt)
        # Note: could be any index since the Ak is the same for all t
        Ak = proj[-1] / (1e-7 + cos[-1])
        return proj, Ak

    def _compute_Bk_coeff(self, X: np.ndarray, t: np.ndarray, dt: float, k: int):
        sin = np.sin(2.0 * np.pi / self.L * k * t)
        proj = function_projection.project_f_onto_g(f=X, g=sin, dx=dt)
        # Note: could be any index since the Bk is the same for all t
        Bk = proj[-1] / (1e-7 + sin[-1])
        return proj, Bk

    def transform(self, X):
        X = np.asarray(X).ravel()
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        X_approx, _ = self._compute_Ak_coeff(X, t, dt, k=0)
        self.A0 = 2.0 * X_approx[0]
        self.Ak, self.Bk = np.zeros((2, self.max_components), dtype=complex)

        for k in np.arange(1, 1 + min((X.size - 2) // 2, self.max_components)):
            proj_cos, Ak = self._compute_Ak_coeff(X, t, dt, k)
            proj_sin, Bk = self._compute_Bk_coeff(X, t, dt, k)

            X_approx += proj_cos + proj_sin

            self.Ak[k - 1] = Ak
            self.Bk[k - 1] = Bk

        return X_approx


class FastFourierApproxC(_FourierApproxCBase):
    def transform(self, X):
        X = np.asarray(X).ravel()
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        upp_lim = min((X.size - 2) // 2, self.max_components // 2)
        low_lim = -upp_lim

        k = np.arange(low_lim, upp_lim + 1)

        basis = np.exp(2.0j * np.pi / self.L * np.outer(k, t))

        # Note: self.L = function_projection.func_norm(basis[k], dt) ** 2, for any k
        self.C = (
            scipy.integrate.trapezoid(X * np.conjugate(basis), dx=dt, axis=1) / self.L
        )

        C = np.expand_dims(self.C, axis=1)

        X_approx = np.sum(C * basis, axis=0)

        return X_approx


class SlowFourierApproxC(_FourierApproxCBase):
    def _compute_projection(self, X: np.ndarray, t: np.ndarray, dt: float, k: int):
        basis = np.exp(1j * 2.0 * np.pi / self.L * k * t)
        proj = function_projection.project_f_onto_g(X, basis, dt)
        # Note: could be any index since the Ck is the same for all t
        Ck = proj[-1] / (1e-7 + basis[-1])
        return proj, Ck

    def transform(self, X):
        X = np.asarray(X).ravel()
        t = np.linspace(0, self.L, X.size)
        dt = t[1] - t[0]

        X_approx = np.zeros_like(X, dtype=complex)

        upp_lim = min((X.size - 2) // 2, self.max_components // 2)
        low_lim = -upp_lim
        self.C = np.zeros(upp_lim - low_lim + 1, dtype=complex)

        for i, k in enumerate(np.arange(low_lim, upp_lim + 1)):
            proj, Ck = self._compute_projection(X, t, dt, k)
            X_approx += proj
            self.C[i] = Ck

        return X_approx


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

    ref_ab = FastFourierApproxAB(L=2 * np.pi)
    ref_ab_slow = SlowFourierApproxAB(L=2 * np.pi)
    ref_c = FastFourierApproxC(L=2 * np.pi)
    ref_c_slow = SlowFourierApproxC(L=2 * np.pi)

    approx_ab = ref_ab.transform(X)
    approx_ab_slow = ref_ab_slow.transform(X)
    approx_c = ref_c.transform(X)
    approx_c_slow = ref_c_slow.transform(X)

    assert np.allclose(approx_ab, approx_ab_slow)
    assert np.allclose(approx_ab, approx_c_slow)
    assert np.allclose(approx_ab, approx_c)
    print(f"RMSE: {np.sqrt(np.mean(np.square(approx_ab[5:-5] - X[5:-5]))):.4f}")

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.set_title("AB Approximation")
    ax1.plot(X, label="original")
    ax1.plot(approx_ab, label="fast")
    ax1.plot(approx_ab_slow, label="slow")
    ax1.legend()

    ax2.set_title("C Approximation")
    ax2.plot(X, label="original")
    ax2.plot(approx_c.real, label="fast")
    ax2.plot(approx_c_slow.real, label="slow")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    _test()
