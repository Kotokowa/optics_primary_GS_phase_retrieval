import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. 读回复场，计算相位
obj_rec = np.load("results_heart_300_v2/reconstructed_object_complex.npy")  # 路径按你的实际来
phase = np.angle(obj_rec)                             # [-pi, pi) 或其它

# 2. 使用和脚本 save_image 一样的 min/max 归一化
vmin = phase.min()
vmax = phase.max()

# 3. 画一条只显示 colorbar 的图（竖向）
fig, ax = plt.subplots(figsize=(1.0, 6))

cmap = plt.get_cmap("viridis")  # 确保和保存图片用的一致
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='vertical'  # 改为竖向
)

cb.set_label("Phase [rad]")
# 可以根据需要设置 tick（这里放 5 个等分）
ticks = np.linspace(vmin, vmax, 5)
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t:.2f}" for t in ticks])

plt.tight_layout()
plt.show()
