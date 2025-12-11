import numpy as np
from scipy.ndimage import gaussian_filter

# Ny = Nx = 300
Ny = 200
Nx = 200
y, x = np.ogrid[-1:1:Ny*1j, -1:1:Nx*1j]

'''
# 心形区域 mask ，边缘平滑过渡
xg = x * 1.3
yg = y * 1.3
f = (xg**2 + yg**2 - 1)**3 - xg**2 * yg**3
mask = f <= 0

# 将布尔 mask 转为浮点并做高斯平滑以获得边缘缓慢过渡
mask_f = mask.astype(np.float32)

# 平滑尺度（像素），可调整，越大过渡越宽
sigma_pixels = 2.0

try:
    blurred = gaussian_filter(mask_f, sigma=sigma_pixels)
except Exception:
    # 如果没有 scipy，使用 FFT 卷积自制高斯核
    Ny, Nx = mask_f.shape
    yy = np.arange(-Ny//2, Ny - Ny//2)
    xx = np.arange(-Nx//2, Nx - Nx//2)
    Xg, Yg = np.meshgrid(xx, yy)
    kernel = np.exp(-(Xg**2 + Yg**2) / (2 * (sigma_pixels**2)))
    kernel /= kernel.sum()
    blurred = np.real(np.fft.ifft2(np.fft.fft2(mask_f) * np.fft.fft2(np.fft.ifftshift(kernel))))

# 归一化到 [0,1]
blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-12)

# 相位：心形内部平滑抬高 0.8 rad，边缘缓慢过渡
phase = (0.8 * blurred).astype(np.float32)
'''

# 心形区域 mask
xg = x * 1.3
yg = y * 1.3
f = (xg**2 + yg**2 - 1)**3 - xg**2 * yg**3
mask = f <= 0

# 相位：一个斜坡 + 心形内部再加一点相位调制
phase = (0.5 * x + 0.3 * y)  # [-O(1), O(1)]
phase += np.where(mask, 0.8, 0.0)   # 心形区域再抬 0.8 rad
phase = phase.astype(np.float32)


# 振幅：心形内部 1，外面 0.2 背景
# amp = np.where(mask, 1.0, 1.0).astype(np.float32)
# amp = (0.5 * x + 1 * y + 1.5)  # [-O(1), O(1)]

# 随机振幅，尺寸保持 Ny x Nx（可加 seed 保证可重复）
amp = np.random.RandomState(0).rand(Ny, Nx).astype(np.float32)

# phase = np.zeros((Ny, Nx), dtype=np.float32)
# amp = (0.8 * blurred).astype(np.float32)

pad_size = 600              # padding 总尺寸从 600 增加到 600+600 = 1200
Ny_pad = Ny + pad_size      # 1200
Nx_pad = Nx + pad_size      # 1200

# 初始化 padded 数组（默认 0）
amp_pad   = np.zeros((Ny_pad, Nx_pad), dtype=np.float32)
phase_pad = np.zeros((Ny_pad, Nx_pad), dtype=np.float32)

# 把原来的 600x600 居中放到 1200x1200 里
sy = (Ny_pad - Ny) // 2      # 起始 y
sx = (Nx_pad - Nx) // 2      # 起始 x

amp_pad[sy:sy+Ny, sx:sx+Nx]   = amp
phase_pad[sy:sy+Ny, sx:sx+Nx] = phase

# 保存 padding 后的 1200x1200 物面
np.save("E:/optics(H)/phase retrieval 1130/my_test_sets_200+200p/my_amp.npy", amp_pad)
np.save("E:/optics(H)/phase retrieval 1130/my_test_sets_200+200p/my_phase.npy", phase_pad)
