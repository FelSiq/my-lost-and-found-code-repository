"""Apply filter using FFT."""
import numpy as np


def _pad(X, pad_size: int):
    X = np.asfarray(X)
    pad = np.zeros(pad_size, dtype=X.dtype)
    X = np.concatenate((pad, X, pad))
    return X


def apply_filter(X, filt, pad: bool = True):
    if pad:
        X = _pad(X, len(filt) // 2)

    X_transf = np.fft.fft(X)
    fil_transf = np.fft.fft(np.flip(filt), n=X.size)

    res = np.real(np.fft.ifft(X_transf * fil_transf))

    if pad:
        res = res[len(filt) - 1 :]

    return res


def _test():
    # Note: mean filter
    filt = [1 / 3, 1 / 3, 1 / 3]
    arr = [1, 2, 10, 3, -1]

    res = apply_filter(arr, filt)

    arr_padded = _pad(arr, len(filt) // 2)
    ref = np.zeros_like(arr_padded)

    for i in range(len(filt), ref.size + 1):
        ref[i - 1] = np.dot(arr_padded[i - len(filt) : i], filt)

    ref = np.real(ref)[len(filt) - 1 :]

    assert np.allclose(ref, res)

    print(res)


if __name__ == "__main__":
    _test()
