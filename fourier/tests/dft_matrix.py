import numpy as np
import matplotlib.pyplot as plt

n = 300


fundamental_freq_n = np.exp(-2.0j * np.pi / n)
dft_mat = (
    np.vander(
        np.power(fundamental_freq_n, np.arange(n)),
        increasing=True,
    )
    / np.sqrt(n)
)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 10), tight_layout=True)

ax1.set_title("Real")
p = ax1.imshow(dft_mat.real)
fig.colorbar(p, ax=ax1)

ax2.set_title("Imag")
p = ax2.imshow(dft_mat.imag)
fig.colorbar(p, ax=ax2)

norm = np.sqrt(np.real(dft_mat * np.conjugate(dft_mat)))
norm = (norm - norm.min()) / (norm.ptp())

ax3.set_title("Normalized norm")
p = ax3.imshow(norm, cmap="ocean")
fig.colorbar(p, ax=ax3)

ax4.set_title("Phase")
p = ax4.imshow(np.arctan(dft_mat.imag / (1e-7 + dft_mat.real)), cmap="magma")
plt.colorbar(p, ax=ax4)

plt.show()
