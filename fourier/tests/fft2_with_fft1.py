import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imglib


img = imglib.imread("img2.jpg")
img = img[::, ::]
img = img.mean(axis=-1).astype(float)
img /= np.max(img)

fft_abs = np.real(np.abs(np.fft.fft2(img, norm="ortho")))
img_fft_row_col, img_fft_col_row = np.empty((2, *img.shape), dtype=complex)

for i in np.arange(img.shape[0]):
    img_fft_row_col[i, :] = np.fft.fft(img[i, :], norm="ortho")

for j in np.arange(img.shape[1]):
    img_fft_row_col[:, j] = np.fft.fft(img_fft_row_col[:, j], norm="ortho")

for j in np.arange(img.shape[1]):
    img_fft_col_row[:, j] = np.fft.fft(img[:, j], norm="ortho")

for i in np.arange(img.shape[0]):
    img_fft_col_row[i, :] = np.fft.fft(img_fft_col_row[i, :], norm="ortho")

img_fft_row_col = np.real(np.abs(img_fft_row_col))
img_fft_col_row = np.real(np.abs(img_fft_col_row))

img_fft_row_col[0, 0] = img_fft_col_row[0, 0] = fft_abs[0, 0] = 0.0
fft_abs = np.fft.fftshift(fft_abs)
img_fft_row_col = np.fft.fftshift(img_fft_row_col)
img_fft_col_row = np.fft.fftshift(img_fft_col_row)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

ax1.imshow(img, cmap="Greys_r")
ax2.imshow(fft_abs, cmap="Greys_r")
ax3.imshow(img_fft_row_col, cmap="Greys_r")
ax4.imshow(img_fft_col_row, cmap="Greys_r")

assert np.allclose(fft_abs, img_fft_row_col)
assert np.allclose(fft_abs, img_fft_col_row)

plt.tight_layout()
plt.show()
