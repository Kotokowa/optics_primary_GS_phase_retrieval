import numpy as np

H, W = 128, 128

# 1. 构造物面振幅：圆形光阑
y, x = np.ogrid[:H, :W]
cy, cx = H / 2, W / 2
r = np.sqrt((y - cy)**2 + (x - cx)**2)
object_amp = (r < H / 4).astype(np.float32)   # 半径 H/4 的圆

# 2. 构造期望的斜面相位（左上 -π，右下 +π）
t = (x + y) / ((W - 1) + (H - 1))  # t 在 [0,1]
phase = -np.pi + 2 * np.pi * t     # 映射到 [-π, π]
phase = phase.astype(np.float32)

# 3. 目标复场
target_complex = object_amp * np.exp(1j * phase)

# 4. 频域振幅（注意：这里不要 fftshift）
F = np.fft.fft2(target_complex)
fourier_amp = np.abs(F).astype(np.float32)

# 5. 保存成 .npy，供你的 gs_phase_retrieval.py 使用
np.save('object_amp.npy', object_amp)
np.save('fourier_amp.npy', fourier_amp)
np.save('target_complex.npy', target_complex.astype(np.complex64))
np.save('phase.npy', phase)  # 可选：保存相位信息