import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


t = np.linspace(-2, 2, 1000)
X = 2 * np.cos(2 * np.pi * 0.5 * t) * np.exp(-t ** 2/2) + np.sin(2 * np.pi * 5 * t + 0.7)
X += np.sin(2 * np.pi * 300 * t)
X[200:250] += t[:50] ** 2
X [100:800] -= 0.5 * t[50:750] ** 3 + 0.5
X += 0.1 * np.random.randn(X.size)

one_sided = False

if one_sided:
    np_fft = np.fft.rfft(X, norm="ortho")
    np_psd = np_fft * np.conjugate(np_fft) * 2
    _, sc_psd = scipy.signal.periodogram(X, detrend="constant", return_onesided=True)

else:
    np_fft = np.fft.fft(X, norm="ortho")
    np_psd = np_fft * np.conjugate(np_fft)
    _, sc_psd = scipy.signal.periodogram(X, detrend="constant", return_onesided=False)

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 10), sharey=True, sharex=True)

ax1.set_title(f"Numpy {'one' if one_sided else 'two'}-sided Power Spectrum Density")
ax1.plot(np_psd.real)

ax2.set_title(f"Scipy {'one' if one_sided else 'two'}-sided Power Spectrum Density")
ax2.plot(sc_psd.real)

ax3.set_title("|Numpy PSD - Scipy PSD|")
ax3.plot(np.abs(np_psd.real - sc_psd.real))

plt.tight_layout()
plt.show()
