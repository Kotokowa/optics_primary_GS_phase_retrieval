import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

# Ny = Nx = 300
Ny = 200
Nx = 300
y, x = np.ogrid[-1:1:Ny*1j, -1:1:Nx*1j]

# ========= 生成 GS 字母掩膜 =========
# 用 matplotlib 的 TextPath 在一个逻辑坐标系里画出 “GS”，再采样到 Ny×Nx 像素
fp = FontProperties(family="DejaVu Sans", weight="bold")
tp = TextPath((0, 0), "GS", size=1.0, prop=fp)  # 这里可以调 size 让字母更粗/更细

bbox = tp.get_extents()  # 字符的边界框
xs = np.linspace(bbox.xmin, bbox.xmax, Nx)
ys = np.linspace(bbox.ymin, bbox.ymax, Ny)
XS, YS = np.meshgrid(xs, ys)

# 判断每个像素中心是否落在 GS 的路径内部
points = np.vstack([XS.ravel(), YS.ravel()]).T
inside = tp.contains_points(points).reshape(Ny, Nx)  # True 的地方就是 GS 字母区域

# ========= 外部不均匀平滑相位（灰度） =========
# 先生成随机噪声，再用高斯滤波平滑，最后归一化到 [-π, π]
noise = np.random.randn(Ny, Nx)
sigma = 8  # 平滑尺度(像素)，可以根据想要的“纹理粗细”调节
smooth_noise = gaussian_filter(noise, sigma=sigma)

# 归一化到 [-π, π]，确保不超过 ±3.14
smooth_noise -= smooth_noise.mean()
smooth_noise /= np.max(np.abs(smooth_noise))
outside_phase = smooth_noise * np.pi  # ∈ [-π, π]

# ========= 组装最终 phase =========
phase = outside_phase.copy()
phase[inside] = 1.57  # GS 字母内部为常数相位 1.57


amp = np.random.RandomState(0).rand(Ny, Nx).astype(np.float32)

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
np.save("E:/optics(H)/phase retrieval 1130/my_test_sets_gs600/my_amp.npy", amp_pad)
np.save("E:/optics(H)/phase retrieval 1130/my_test_sets_gs600/my_phase.npy", phase_pad)