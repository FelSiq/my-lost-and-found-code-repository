import typing as t

import matplotlib.pyplot as plt
import numpy as np


def mandelbrot(lim_real: t.Tuple[float, float] = (-2.0, 1.0),
               lim_img: t.Tuple[float, float] = (-1.0, 1.0),
               num_points_real: int = 1280,
               num_points_img: int = 1028,
               num_it: int = 128,
               plot: bool = False,
               func_color: t.Optional[t.Callable[[int], int]] = None,
               **kwargs) -> np.ndarray:
    """Generate the mandelbrot set in the given range."""
    vals_real = np.linspace(*lim_real, num_points_real)
    vals_img = np.linspace(*lim_img, num_points_img)

    R, I = np.meshgrid(vals_real, vals_img)

    _dtype = np.uint8 if num_it <= 256 else np.uint32

    colors = np.zeros((num_points_img, num_points_real), dtype=_dtype)

    _it_range = np.arange(num_it, dtype=_dtype)

    _it_cur = 0
    _it_tot = colors.size

    if func_color is None:

        def func_color(it_num: int) -> int:
            return it_num

    for ind_r in np.arange(num_points_real):
        for ind_i in np.arange(num_points_img):
            c = complex(R[ind_i, ind_r], I[ind_i, ind_r])
            z = 0.0 + 0.0j

            for ind_color in _it_range:
                if abs(z) > 2:
                    colors[ind_i, ind_r] = func_color(ind_color)
                    break

                z = z**2 + c

            _it_cur += 1

        if _it_cur % 100 == 0:
            print("\r{:.2f}%".format(100 * _it_cur / _it_tot), end="")

    print("\rDone.")

    if plot:
        plt.suptitle("Mandelbrot set")
        plt.title("Real range: {} - Img range: {} - Iterations: {}".format(
            lim_real, lim_img, num_it))
        plt.imshow(colors / np.max(colors), **kwargs)
        plt.xlabel("Real axis")
        plt.ylabel("Imaginary axis")
        plt.show()

    return colors


def _test() -> None:
    mandelbrot(plot=True, func_color=None, cmap="Spectral")


if __name__ == "__main__":
    _test()
