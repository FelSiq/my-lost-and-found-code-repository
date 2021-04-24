import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime


def draw(X, coeffs, xlim=(0, 1), ylim=(0, 1), interval: int = 5, output_file=None):
    def update(j):
        center = complex()

        for i, c in enumerate(coeffs[j]):
            new_center = center + c
            x = center.real, new_center.real
            y = center.imag, new_center.imag
            line = artists[1 + 2 * i]
            line.set_data(x, y)

            center = new_center
            circle_center = center.real, center.imag
            circle_radius = np.real(np.abs(c))
            circle = artists[1 + 2 * i + 1]
            circle.center = circle_center
            circle.set_radius(circle_radius)

        cur_points = approx[: j + 1]
        aux.set_data(cur_points.real, cur_points.imag)

        return artists

    fig, ax = plt.subplots(1, figsize=(10, 10))

    approx = np.sum(coeffs, axis=1)

    ax.set_title(f"Fourier series approximation (number of components: {len(coeffs)})")
    ax.plot(X.real, X.imag, linestyle="dotted", label="original")
    (aux,) = ax.plot(approx.real, approx.imag, label="fourier approximation")
    ax.legend(loc="upper right")

    artists = [aux]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    center = complex()

    for c in coeffs[0]:
        new_center = center + c
        x = center.real, new_center.real
        y = center.imag, new_center.imag
        (line,) = ax.plot(x, y)

        center = new_center
        circle_center = center.real, center.imag
        circle_radius = np.real(np.abs(c))
        circle = plt.Circle(circle_center, circle_radius, fill=False)
        ax.add_patch(circle)

        artists.append(line)
        artists.append(circle)

    ani = anime.FuncAnimation(
        fig,
        update,
        interval=interval,
        frames=np.arange(len(coeffs)),
        blit=True,
        repeat=False,
        cache_frame_data=True,
    )

    if output_file is not None:
        if not output_file.endswith(".mp4"):
            output_file += ".mp4"

        print(f"Saving into output file ({output_file})...")

        ani.save(output_file, writer="ffmpeg", fps=30)

    plt.show()


def _test():
    def gen_coeffs():
        return np.random.randn(10) + 1j * np.random.randn(10)

    def interpolate(coeffs_a, coeffs_b):
        real_a = coeffs_a.real
        real_b = coeffs_b.real
        imag_a = coeffs_a.imag
        imag_b = coeffs_b.imag

        real_interpol = (real_b - real_a) * t + real_a
        imag_interpol = (imag_b - imag_a) * t + imag_a

        return real_interpol + 1j * imag_interpol

    t = np.linspace(0, 1, 100).reshape(-1, 1)

    first_coeffs = gen_coeffs()
    coeffs = [first_coeffs]

    for i in range(2):
        last_coeffs = gen_coeffs()
        coeffs.append(interpolate(coeffs[-1][-1], last_coeffs))

    coeffs.append(interpolate(last_coeffs, first_coeffs))
    coeffs = np.vstack(coeffs)

    draw(coeffs)


if __name__ == "__main__":
    _test()
