import numpy as np
import imageio.v2 as iio

amp_img  = iio.imread("amp.png").astype(np.float32)
phi_img  = iio.imread("phi.png").astype(np.float32)

# 振幅归一化到你想要的范围，比如 [0,1]
amp = amp_img
amp -= amp.min()
if amp.max() > 0:
    amp /= amp.max()

# 相位映射到 [0, 2π]
phase = (phi_img / 255.0) * (2 * np.pi)

np.save("my_amp.npy", amp)
np.save("my_phase.npy", phase)
