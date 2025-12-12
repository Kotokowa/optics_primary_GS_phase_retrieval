import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

# ---------------- 基本参数 ----------------
Ny = 200
Nx = 400
y, x = np.ogrid[-1:1:Ny*1j, -1:1:Nx*1j]

# 保存路径（确保这个路径存在或可创建）
save_dir = r"E:/optics(H)/gs_phase_retrieval_with_64_tests_11_20_new/my_test_sets_121102"
os.makedirs(save_dir, exist_ok=True)

# ---------------- 1. 生成“光學”文字掩膜 ----------------
# PIL 图像的尺寸顺序是 (宽, 高) = (Nx, Ny)
img = Image.new("L", (Nx, Ny), 0)  # 黑底
draw = ImageDraw.Draw(img)

text = "光學"
font_size = int(min(Nx, Ny) * 0.7)

# 尝试加载常见的中文字体
font = None
font_candidates = [
    "simhei.ttf",
    "simsun.ttc",
    "msyh.ttc",
    "NotoSansCJK-Regular.ttc",
    "Arial Unicode.ttf",
]
for f in font_candidates:
    try:
        font = ImageFont.truetype(f, font_size)
        break
    except IOError:
        continue

# 如果上面都没找到，就用默认字体（可能不太好看，但能跑）
if font is None:
    font = ImageFont.load_default()

# 计算文字居中位置
bbox = draw.textbbox((0, 0), text, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
text_x = (Nx - text_w) // 2
text_y = (Ny - text_h) // 2

# 写白色“光學”（255），背景为 0
draw.text((text_x, text_y), text, fill=255, font=font)

# 转成 NumPy 掩膜：文字区域为 True
mask = np.array(img, dtype=np.float32) / 255.0
char_region = mask > 0.5

# ---------------- 2. 生成外部平滑随机相位 ----------------
np.random.seed(0)  # 如果不需要可重复性，可以注释掉
phase_bg = np.random.uniform(-np.pi, np.pi, size=(Ny, Nx))

# 高斯平滑让外部相位更平滑
phase_bg_smooth = gaussian_filter(phase_bg, sigma=10)

# 平移到零均值，并归一化到 [-π, π] 内，保证不超过 ±3.14
phase_bg_smooth -= phase_bg_smooth.mean()
max_abs = np.max(np.abs(phase_bg_smooth))
if max_abs > 0:
    phase_bg_smooth = phase_bg_smooth / max_abs * np.pi

# ---------------- 3. 组合“光學”内部和外部相位 ----------------
phase = phase_bg_smooth.copy()
phase[char_region] = -3.00  # 文字内部固定

print("phase 原始尺寸:", phase.shape)  # 期望 (200, 400)

# ---------------- 4. 加 1000 像素 0 padding（每一边）并保存 phase.npy ----------------
pad = 1000
# 显式指定 (前后, 前后)，避免任何歧义
pad_width = ((pad, pad), (pad, pad))

phase_padded = np.pad(
    phase,
    pad_width=pad_width,
    mode="constant",
    constant_values=0.0
)

print("phase 带 padding 尺寸:", phase_padded.shape)  # 期望 (2200, 2400)

phase_path = os.path.join(save_dir, "phase.npy")
np.save(phase_path, phase_padded)
print("phase.npy 已保存到:", phase_path)

# ---------------- 5. 生成随机幅度 amp（200×400），同样 padding 并保存 amp.npy ----------------
amp = np.random.rand(Ny, Nx)  # 0~1 随机幅度
print("amp 原始尺寸:", amp.shape)  # 期望 (200, 400)

amp_padded = np.pad(
    amp,
    pad_width=pad_width,
    mode="constant",
    constant_values=0.0
)

print("amp 带 padding 尺寸:", amp_padded.shape)  # 期望 (2200, 2400)

amp_path = os.path.join(save_dir, "amp.npy")
np.save(amp_path, amp_padded)
print("amp.npy 已保存到:", amp_path)
