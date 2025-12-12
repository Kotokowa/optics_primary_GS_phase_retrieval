import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


def fft_gaussian_blur(img, sigma_pix):
    if sigma_pix <= 0:
        return img.astype(np.float64)
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-2 * (np.pi**2) * (sigma_pix**2) * (FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(img) * H).real


def smooth_noise(shape, sigma_pix=10, seed=0, amp=0.15):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape)
    n = fft_gaussian_blur(n, sigma_pix)
    n = n / (np.std(n) + 1e-12)
    return amp * n


def wrap_to_pi(phase):
    return (phase + np.pi) % (2 * np.pi) - np.pi


def create_xy_grid(N, pixel_size):
    x = (np.arange(N) - N // 2) * pixel_size
    X, Y = np.meshgrid(x, x)
    return X, Y


def rounded_rect_window(X, Y, wx, wy, edge=0.08, p=6.0):
    ax = np.abs(X) / (wx + 1e-12)
    ay = np.abs(Y) / (wy + 1e-12)
    r = (ax**p + ay**p) ** (1 / p)

    t0 = 1.0 - edge
    t1 = 1.0
    w = np.ones_like(r)
    w[r >= t1] = 0.0
    mid = (r >= t0) & (r < t1)
    s = (r[mid] - t0) / (t1 - t0 + 1e-12)
    w[mid] = 1 - (3 * s**2 - 2 * s**3)  # smoothstep
    return w


def rasterize_text_mask(N, text, center_xy_pix, font_size, weight="bold"):
    fp = FontProperties(family="DejaVu Sans", weight=weight)
    tp = TextPath((0, 0), text, size=font_size, prop=fp)
    verts = tp.vertices.copy()

    # center the glyph around origin
    minv = verts.min(axis=0)
    maxv = verts.max(axis=0)
    w = maxv[0] - minv[0]
    h = maxv[1] - minv[1]
    verts[:, 0] -= (minv[0] + w / 2)
    verts[:, 1] -= (minv[1] + h / 2)

    cx, cy = center_xy_pix
    px = verts[:, 0] + cx
    py = -verts[:, 1] + cy  # flip y for image coords

    from matplotlib.path import Path
    path = Path(np.vstack([px, py]).T, tp.codes)

    yy, xx = np.mgrid[0:N, 0:N]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    inside = path.contains_points(pts)
    return inside.reshape(N, N).astype(np.float64)


def make_snellen_soft_phase(N, blur_sigma=2.0, seed=1, phase_scale=2.6*np.pi):
    canvas = np.zeros((N, N), dtype=np.float64)
    rows = [("E", 140), ("FP", 110), ("TOZ", 90), ("LPED", 75), ("PECFD", 60)]
    y0 = int(N * 0.22)
    dy = int(N * 0.14)
    for i, (txt, fs) in enumerate(rows):
        cy = y0 + i * dy
        cx = N // 2
        m = rasterize_text_mask(N, txt, (cx, cy), font_size=fs, weight="bold")
        canvas = np.maximum(canvas, m)

    smooth = fft_gaussian_blur(canvas, blur_sigma)
    smooth = smooth / (smooth.max() + 1e-12)
    smooth += smooth_noise((N, N), sigma_pix=12, seed=seed, amp=0.10)
    smooth = np.clip(smooth, 0.0, 1.0)
    return phase_scale * smooth


def make_optics_soft_phase(N, blur_sigma=2.5, seed=2, font_size=130, phase_scale=2.8*np.pi):
    # 关键：减小字号 + 稍微右移，避免被支持域裁掉（你之前 target 里 "o" 缺了一块）
    cx = int(N * 0.50)
    cy = int(N * 0.50)

    m = rasterize_text_mask(N, "optics", (cx, cy), font_size=80, weight="bold")
    smooth = fft_gaussian_blur(m, blur_sigma)
    smooth = smooth / (smooth.max() + 1e-12)

    # 字母内部加一点点局部倾斜（非二值）
    yy, xx = np.mgrid[0:N, 0:N]
    tilt_local = (xx - cx) / (N + 1e-12) * 0.25 + (yy - cy) / (N + 1e-12) * (-0.18)
    smooth2 = np.clip(smooth + 0.20 * smooth * tilt_local, 0.0, 1.0)

    smooth2 += smooth_noise((N, N), sigma_pix=14, seed=seed, amp=0.10)
    smooth2 = np.clip(smooth2, 0.0, 1.0)
    return phase_scale * smooth2


def main():
    parser = argparse.ArgumentParser(description="Program 1: generate object complex/intensity/phase/mask")
    parser.add_argument("--mode", choices=["snellen", "optics"], default="snellen")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--pixel_size", type=float, default=6e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")

    # 控制强度/相位“难度”
    parser.add_argument("--tilt_strength", type=float, default=0.5*np.pi, help="global tilt strength (rad)")
    parser.add_argument("--phase_noise_amp", type=float, default=0.08*np.pi, help="global phase noise amp (rad)")
    parser.add_argument("--texture_amp", type=float, default=0.10, help="intensity texture amp")
    args = parser.parse_args()

    N = args.N
    ps = args.pixel_size
    seed = args.seed
    mode = args.mode
    prefix = args.prefix if args.prefix is not None else f"{mode}_"

    X, Y = create_xy_grid(N, ps)

    # 支持域：圆角矩形软窗 + 二值 mask（GS 用二值）
    wx = (N * ps) * 0.35
    wy = (N * ps) * 0.35
    win = rounded_rect_window(X, Y, wx=wx, wy=wy, edge=0.08)
    mask = (win > 0.05).astype(np.float64)

    # 强度：支持域内有纹理，但不要太强（纹理太强会让某些 case 更容易卡局部解）
    I_base = (0.25 + 0.75 * win) ** 2
    tex = fft_gaussian_blur(np.random.default_rng(seed).random((N, N)), 10)
    tex = (tex - tex.mean()) / (tex.std() + 1e-12)
    I_texture = 1.0 + args.texture_amp * tex
    intensity = I_base * I_texture
    intensity = np.clip(intensity, 0.0, None)
    intensity *= mask

    # 相位主体
    if mode == "snellen":
        phase_main = make_snellen_soft_phase(N, blur_sigma=2.0, seed=seed+1)
    else:
        phase_main = make_optics_soft_phase(N, blur_sigma=2.5, seed=seed+2, font_size=135)

    # 全局 tilt + 平滑噪声（轻一点：optics 更稳）
    tilt = args.tilt_strength * (
        0.6 * X / (np.max(np.abs(X)) + 1e-12) + 0.4 * Y / (np.max(np.abs(Y)) + 1e-12)
    )
    ph_noise = args.phase_noise_amp * smooth_noise((N, N), sigma_pix=18, seed=seed+3, amp=1.0)

    phase = (phase_main + tilt + ph_noise) * mask
    phase = wrap_to_pi(phase)

    obj_complex = np.sqrt(np.maximum(intensity, 0.0)) * np.exp(1j * phase)

    np.save(prefix + "complex.npy", obj_complex)
    np.save(prefix + "intensity.npy", intensity)
    np.save(prefix + "phase.npy", phase)
    np.save(prefix + "mask.npy", mask)

    print("Saved:")
    print(" ", prefix + "complex.npy")
    print(" ", prefix + "intensity.npy")
    print(" ", prefix + "phase.npy")
    print(" ", prefix + "mask.npy")

    if args.show:
        vmin, vmax = -3.15, 3.15
        cmap = plt.get_cmap("twilight").copy()
        cmap.set_bad(color="black")

        ph_show = phase.copy()
        ph_show[mask < 0.5] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        im0 = axs[0].imshow(intensity, cmap="gray")
        axs[0].set_title("Object intensity")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(ph_show, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Object phase ({mode}) [rad]")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(mask, cmap="gray")
        axs[2].set_title("Support mask")
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
