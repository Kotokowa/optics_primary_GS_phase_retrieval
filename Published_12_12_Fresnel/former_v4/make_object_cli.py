import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path


def fft_gaussian_blur(img, sigma_pix):
    if sigma_pix <= 0:
        return img.astype(np.float64)
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-2 * (np.pi**2) * (sigma_pix**2) * (FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(img) * H).real


def smooth_noise(shape, sigma_pix=12, seed=0, amp=1.0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape)
    n = fft_gaussian_blur(n, sigma_pix)
    n = n / (np.std(n) + 1e-12)
    return amp * n


def wrap_to_pi(ph):
    return (ph + np.pi) % (2 * np.pi) - np.pi


def rasterize_text_mask(N, text, center_xy_pix, font_size, weight="bold"):
    fp = FontProperties(family="DejaVu Sans", weight=weight)
    tp = TextPath((0, 0), text, size=font_size, prop=fp)

    verts = tp.vertices.copy()
    minv = verts.min(axis=0)
    maxv = verts.max(axis=0)
    w = maxv[0] - minv[0]
    h = maxv[1] - minv[1]
    verts[:, 0] -= (minv[0] + w / 2)
    verts[:, 1] -= (minv[1] + h / 2)

    cx, cy = center_xy_pix
    px = verts[:, 0] + cx
    py = -verts[:, 1] + cy

    path = Path(np.vstack([px, py]).T, tp.codes)

    yy, xx = np.mgrid[0:N, 0:N]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    inside = path.contains_points(pts)
    return inside.reshape(N, N).astype(np.float64)


def rounded_rect_soft_window(N, edge=0.08, p=6.0):
    """软窗（幅度用），减少硬边带来的边界伪影"""
    yy, xx = np.mgrid[0:N, 0:N]
    x = (xx - N/2) / (0.35 * N)
    y = (yy - N/2) / (0.35 * N)
    r = (np.abs(x)**p + np.abs(y)**p) ** (1/p)

    t0 = 1.0 - edge
    t1 = 1.0
    w = np.ones_like(r)
    w[r >= t1] = 0.0
    mid = (r >= t0) & (r < t1)
    s = (r[mid] - t0) / (t1 - t0 + 1e-12)
    w[mid] = 1 - (3*s**2 - 2*s**3)  # smoothstep
    return w


def make_snellen_phase(N, seed=0, phase_scale=2.6*np.pi):
    canvas = np.zeros((N, N), dtype=np.float64)
    rows = [("E", 140), ("FP", 110), ("TOZ", 90), ("LPED", 75), ("PECFD", 60)]
    y0 = int(N * 0.22)
    dy = int(N * 0.14)
    for i, (txt, fs) in enumerate(rows):
        canvas = np.maximum(canvas, rasterize_text_mask(N, txt, (N//2, y0 + i*dy), fs))
    smooth = fft_gaussian_blur(canvas, 2.0)
    smooth = smooth / (smooth.max() + 1e-12)
    smooth = np.clip(smooth + 0.10 * smooth_noise((N, N), 14, seed=seed+1), 0.0, 1.0)
    return phase_scale * smooth


def make_optics_phase(N, seed=0, font_size=80, phase_scale=2.8*np.pi):
    # 重要：不要靠边，确保完全在支持域里
    cx = int(N * 0.54)
    cy = int(N * 0.50)
    m = rasterize_text_mask(N, "optics", (cx, cy), font_size, weight="bold")
    smooth = fft_gaussian_blur(m, 2.5)
    smooth = smooth / (smooth.max() + 1e-12)

    # 非二值：字母内部轻微倾斜 + 少量噪声（别太大）
    yy, xx = np.mgrid[0:N, 0:N]
    tilt_local = (xx - cx) / (N + 1e-12) * 0.18 + (yy - cy) / (N + 1e-12) * (-0.14)
    smooth2 = np.clip(smooth + 0.18 * smooth * tilt_local, 0.0, 1.0)
    smooth2 = np.clip(smooth2 + 0.08 * smooth_noise((N, N), 18, seed=seed+2), 0.0, 1.0)
    return phase_scale * smooth2


def main():
    ap = argparse.ArgumentParser("Program1: create object for Fresnel GS")
    ap.add_argument("--mode", choices=["snellen", "optics"], default="snellen")
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--support_thresh", type=float, default=0.05, help="threshold to binarize support mask from soft window")
    ap.add_argument("--tilt_strength", type=float, default=0.30*np.pi)
    ap.add_argument("--phase_noise_amp", type=float, default=0.05*np.pi)

    ap.add_argument("--optics_font", type=int, default=80)
    args = ap.parse_args()

    N = args.N
    seed = args.seed
    prefix = args.prefix if args.prefix is not None else f"{args.mode}_"

    win = rounded_rect_soft_window(N, edge=0.08)
    support_mask = (win > args.support_thresh).astype(np.float64)

    # 物面幅度：用软窗（稳定），支持域外严格为0
    amp = (0.25 + 0.75 * win) * support_mask
    intensity = amp**2

    # 相位主体
    if args.mode == "snellen":
        phase_main = make_snellen_phase(N, seed=seed)
    else:
        phase_main = make_optics_phase(N, seed=seed, font_size=args.optics_font)

    # 全局 tilt + 平滑相位噪声（不要太大）
    yy, xx = np.mgrid[0:N, 0:N]
    x = (xx - N/2) / (N/2 + 1e-12)
    y = (yy - N/2) / (N/2 + 1e-12)
    tilt = args.tilt_strength * (0.6*x + 0.4*y)
    ph_noise = args.phase_noise_amp * smooth_noise((N, N), 22, seed=seed+10)

    phase = (phase_main + tilt + ph_noise) * support_mask
    phase = wrap_to_pi(phase)

    obj = amp * np.exp(1j * phase)

    np.save(prefix + "complex.npy", obj)
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

        ph_show = phase.copy()
        ph_show[support_mask < 0.5] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        im0 = axs[0].imshow(intensity, cmap="gray")
        axs[0].set_title("Object intensity")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(ph_show, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title("Object phase [rad]")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(support_mask, cmap="gray")
        axs[2].set_title("Support mask")
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
