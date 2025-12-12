import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


def create_xy_grid(N, pixel_size):
    x = (np.arange(N) - N // 2) * pixel_size
    X, Y = np.meshgrid(x, x)
    return X, Y


def fft_gaussian_blur(img, sigma_pix):
    if sigma_pix <= 0:
        return img.astype(np.float64)
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-2 * (np.pi**2) * (sigma_pix**2) * (FX**2 + FY**2))
    out = np.fft.ifft2(np.fft.fft2(img) * H).real
    return out


def smooth_noise(N, sigma_pix=6, seed=0, amp=0.15):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((N, N))
    n = fft_gaussian_blur(n, sigma_pix)
    n = n / (np.std(n) + 1e-12)
    return amp * n


def rounded_rect_window(X, Y, wx, wy, edge=0.08):
    ax = np.abs(X) / (wx + 1e-12)
    ay = np.abs(Y) / (wy + 1e-12)
    p = 6.0
    r = (ax**p + ay**p)**(1/p)

    t0 = 1.0 - edge
    t1 = 1.0
    w = np.ones_like(r)
    w[r >= t1] = 0.0
    mid = (r >= t0) & (r < t1)
    s = (r[mid] - t0) / (t1 - t0 + 1e-12)
    w[mid] = 1 - (3*s**2 - 2*s**3)
    return w


def rasterize_text_to_mask(N, text, center_xy_pix, font_size, weight="bold"):
    fp = FontProperties(family="DejaVu Sans", weight=weight)
    tp = TextPath((0, 0), text, size=font_size, prop=fp)

    verts = tp.vertices.copy()
    minv = verts.min(axis=0)
    maxv = verts.max(axis=0)
    w = maxv[0] - minv[0]
    h = maxv[1] - minv[1]
    verts[:, 0] -= (minv[0] + w/2)
    verts[:, 1] -= (minv[1] + h/2)

    cx, cy = center_xy_pix
    px = verts[:, 0] + cx
    py = -verts[:, 1] + cy

    from matplotlib.path import Path
    path = Path(np.vstack([px, py]).T, tp.codes)

    yy, xx = np.mgrid[0:N, 0:N]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    inside = path.contains_points(pts)
    mask = inside.reshape(N, N).astype(np.float64)
    return mask


def make_snellen_phase(N, blur_sigma=2.0, seed=1):
    canvas = np.zeros((N, N), dtype=np.float64)
    rows = [
        ("E", 140),
        ("FP", 110),
        ("TOZ", 90),
        ("LPED", 75),
        ("PECFD", 60),
    ]
    y0 = int(N * 0.22)
    dy = int(N * 0.14)
    for i, (txt, fs) in enumerate(rows):
        cy = y0 + i * dy
        cx = N // 2
        m = rasterize_text_to_mask(N, txt, (cx, cy), font_size=fs, weight="bold")
        canvas = np.maximum(canvas, m)

    smooth = fft_gaussian_blur(canvas, blur_sigma)
    smooth = smooth / (smooth.max() + 1e-12)
    smooth += smooth_noise(N, sigma_pix=10, seed=seed, amp=0.10)
    smooth = np.clip(smooth, 0.0, 1.0)
    phase = (2.6 * np.pi) * smooth
    return phase


def make_optics_phase(N, blur_sigma=2.5, seed=2):
    cx, cy = N // 2, N // 2
    m = rasterize_text_to_mask(N, "optics", (cx, cy), font_size=160, weight="bold")
    smooth = fft_gaussian_blur(m, blur_sigma)
    smooth = smooth / (smooth.max() + 1e-12)

    yy, xx = np.mgrid[0:N, 0:N]
    tilt_local = (xx - cx) / (N + 1e-12) * 0.35 + (yy - cy) / (N + 1e-12) * (-0.20)
    smooth2 = np.clip(smooth + 0.25 * smooth * tilt_local, 0.0, 1.0)

    smooth2 += smooth_noise(N, sigma_pix=12, seed=seed, amp=0.10)
    smooth2 = np.clip(smooth2, 0.0, 1.0)
    phase = (2.8 * np.pi) * smooth2
    return phase


def wrap_to_pi(phase):
    return (phase + np.pi) % (2*np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser(description="Program 1: generate object complex/intensity/phase/mask")
    parser.add_argument("--mode", choices=["snellen", "optics"], default="snellen", help="phase pattern")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--pixel_size", type=float, default=6e-6, help="meters")
    parser.add_argument("--prefix", type=str, default=None, help="output prefix, e.g. snellen_")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="show figures")
    args = parser.parse_args()

    N = args.N
    pixel_size = args.pixel_size
    mode = args.mode
    seed = args.seed

    prefix = args.prefix if args.prefix is not None else f"{mode}_"

    X, Y = create_xy_grid(N, pixel_size)

    # 非圆 support：圆角矩形软窗
    wx = (N * pixel_size) * 0.35
    wy = (N * pixel_size) * 0.35
    edge_soft = 0.08

    win = rounded_rect_window(X, Y, wx=wx, wy=wy, edge=edge_soft)
    support_mask = (win > 0.05).astype(np.float64)

    # 强度（支持域内非均匀 + 支持域外严格 0）
    I_base = (0.3 + 0.7 * win)**2
    I_texture = 1.0 + 0.12 * fft_gaussian_blur(np.random.default_rng(seed).random((N, N)), 10)
    intensity = I_base * I_texture
    intensity *= support_mask

    # 相位主体
    if mode == "snellen":
        phase_main = make_snellen_phase(N, blur_sigma=2.0, seed=seed+1)
    else:
        phase_main = make_optics_phase(N, blur_sigma=2.5, seed=seed+2)

    # 全局 tilt + 平滑噪声（背景也有）
    global_tilt_strength = 0.8 * np.pi
    global_noise_amp = 0.10 * np.pi
    noise_sigma_pix = 14

    tilt = global_tilt_strength * (0.6 * X / (np.max(np.abs(X)) + 1e-12) +
                                   0.4 * Y / (np.max(np.abs(Y)) + 1e-12))
    ph_noise = global_noise_amp * smooth_noise(N, sigma_pix=noise_sigma_pix, seed=seed+3, amp=1.0)

    phase = (phase_main + tilt + ph_noise) * support_mask
    phase = wrap_to_pi(phase)

    amplitude = np.sqrt(np.maximum(intensity, 0.0))
    obj_complex = amplitude * np.exp(1j * phase)

    np.save(prefix + "complex.npy", obj_complex)
    np.save(prefix + "intensity.npy", intensity)
    np.save(prefix + "phase.npy", phase)
    np.save(prefix + "mask.npy", support_mask)

    print("Saved:")
    print(" ", prefix + "complex.npy")
    print(" ", prefix + "intensity.npy")
    print(" ", prefix + "phase.npy")
    print(" ", prefix + "mask.npy")

    if args.show:
        vmin, vmax = -3.15, 3.15
        cmap = plt.get_cmap("twilight").copy()
        cmap.set_bad(color="black")
        phase_show = phase.copy()
        phase_show[support_mask < 0.5] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        im0 = axs[0].imshow(intensity, cmap="gray")
        axs[0].set_title("Object intensity")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(phase_show, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Object phase ({mode}) [rad]")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(support_mask, cmap="gray")
        axs[2].set_title("Support mask")
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
