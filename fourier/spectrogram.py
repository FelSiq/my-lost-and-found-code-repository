import numpy as np
import scipy.stats


def _adjust_filter(filt, window_size: int):
    if window_size == filt.size:
        return filt

    f_size = window_size / 2

    filt_window = filt[
        filt.size // 2 - int(f_size) : filt.size // 2 + int(np.ceil(f_size))
    ]

    return filt_window


def gabor_transform(X, window_size: int = 256, noverlap: int = 32):
    filt = scipy.signal.tukey(window_size + 1, alpha=0.25)[:-1]

    n_windows = X.size // (window_size - noverlap)
    spectrogram = np.empty((window_size, n_windows), dtype=complex)

    for i in np.arange(n_windows):
        start_ind = i * (window_size - noverlap)
        end_ind = start_ind + window_size

        X_window = X[start_ind:end_ind]
        filt_window = _adjust_filter(filt, X_window.size)
        X_window_weighted = X_window * filt_window
        X_window_t = np.fft.fft(X_window_weighted, n=window_size)

        spectrogram[:, i] = X_window_t

    spectrogram *= 1.0 / np.sqrt(np.sum(np.square(filt)))
    spectrogram = np.abs(spectrogram)

    return spectrogram


def _test():
    import scipy.signal
    import matplotlib.pyplot as plt

    t = np.linspace(-2, 2, 1000)
    X = (
        3 * np.cos(2 * np.pi * 0.5 * t) * np.exp(-(t ** 2) / 2)
        + (-50 * t ** 2 + 5 * np.sin(2 * np.pi * 5 * t)) * np.exp(-(t ** 2) / 0.5)
        + 1
    )
    X += 0.5 * np.random.randn(X.size)

    window_size = 8
    noverlap = 4

    f, _, Sxx = scipy.signal.spectrogram(
        X,
        nperseg=window_size,
        noverlap=noverlap,
        return_onesided=False,
        window=("tukey", 0.25),
        detrend=False,
        mode="magnitude",
    )
    spectrogram = gabor_transform(X, window_size=window_size, noverlap=noverlap)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15, 10))
    ax1.set_title("Data")
    ax1.plot(X)

    min_ = min(np.min(Sxx), np.min(spectrogram))
    max_ = max(np.max(Sxx), np.max(spectrogram))

    ax2.set_title("Reference (from Scipy)")
    p = ax2.imshow(Sxx, cmap="magma", aspect="auto", vmin=min_, vmax=max_)
    fig.colorbar(p, ax=ax2, orientation="horizontal", fraction=0.05, aspect=150)

    ax3.set_title("Spectrogram (implemented)")
    p = ax3.imshow(spectrogram, cmap="magma", aspect="auto", vmin=min_, vmax=max_)
    fig.colorbar(p, ax=ax3, orientation="horizontal", fraction=0.05, aspect=150)

    ax4.set_title("Diff (|Reference - Implemented|)")
    p = ax4.imshow(
        np.abs(Sxx - spectrogram[:, : Sxx.shape[1]]),
        cmap="gray",
        aspect="auto",
        vmin=min_,
        vmax=max_,
    )
    fig.colorbar(p, ax=ax4, orientation="horizontal", fraction=0.05, aspect=150)

    print(Sxx.shape, spectrogram.shape)
    print(np.abs(Sxx).max(), np.abs(Sxx).min())
    print(np.abs(spectrogram).max(), np.abs(spectrogram).min())
    print(np.abs(Sxx - spectrogram[:, : Sxx.shape[1]]).max())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _test()
