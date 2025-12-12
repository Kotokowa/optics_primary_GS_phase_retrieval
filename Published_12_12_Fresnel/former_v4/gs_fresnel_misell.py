import numpy as np
import matplotlib.pyplot as plt
import argparse


def fresnel_propagation_ASM(field: np.ndarray, wavelength: float, pixel_size: float, z: float) -> np.ndarray:
    k = 2 * np.pi / wavelength
    ny, nx = field.shape
    dx = dy = pixel_size

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    fsq = FX**2 + FY**2
    arg = 1.0 - (wavelength**2) * fsq

    # 允许复数传播（含倏逝波）。如果你想更稳，可改成 np.maximum(arg,0) 抑制倏逝波
    H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))
    return np.fft.ifft2(np.fft.fft2(field) * H)


def check_escape_pressed() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False
    if msvcrt.kbhit():
        return msvcrt.getch() == b"\x1b"
    return False


def nmse_intensity(Ip: np.ndarray, I: np.ndarray) -> float:
    return float(np.mean((Ip - I) ** 2) / (np.mean(I ** 2) + 1e-30))


def nmse_complex(gt: np.ndarray, rec: np.ndarray, mask: np.ndarray | None = None) -> float:
    gt = gt.astype(np.complex128)
    rec = rec.astype(np.complex128)
    if mask is not None:
        gt = gt[mask]
        rec = rec[mask]
    alpha = np.vdot(rec, gt) / (np.vdot(rec, rec) + 1e-30)
    diff = gt - alpha * rec
    return float(np.mean(np.abs(diff) ** 2) / (np.mean(np.abs(gt) ** 2) + 1e-30))


def align_global_phase(rec: np.ndarray, ref: np.ndarray, mask_bool: np.ndarray) -> tuple[np.ndarray, float]:
    r = rec[mask_bool]
    t = ref[mask_bool]
    alpha = np.vdot(r, t) / (np.vdot(r, r) + 1e-30)
    phase_factor = alpha / (np.abs(alpha) + 1e-30)
    return rec * phase_factor, float(np.angle(phase_factor))


def circular_mean_phase(field: np.ndarray, mask_bool: np.ndarray) -> float:
    z = field[mask_bool]
    s = np.sum(z / (np.abs(z) + 1e-30))
    return float(np.angle(s + 1e-30))


def relax_amp(field: np.ndarray, amp_target: np.ndarray, gamma: float) -> np.ndarray:
    if gamma >= 1.0:
        return amp_target * np.exp(1j * np.angle(field))
    return (1 - gamma) * field + gamma * (amp_target * np.exp(1j * np.angle(field)))


def run_single_plane(
    I_obj: np.ndarray,
    I1: np.ndarray,
    mask: np.ndarray,
    wl: float,
    dx: float,
    z1: float,
    init: str,
    max_iters: int,
    gamma_obj: float,
    gamma_sen: float,
    target_conv: float | None,
    allow_esc: bool,
    fix_piston: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp_obj = np.sqrt(np.maximum(I_obj, 0.0))
    amp1 = np.sqrt(np.maximum(I1, 0.0))

    ny, nx = I_obj.shape
    if init == "random":
        rng = np.random.default_rng(1234)
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    else:
        phase0 = np.zeros((ny, nx), dtype=np.float64)

    obj = amp_obj * np.exp(1j * phase0)
    obj *= mask

    mask_bool = mask > 0.5
    errors: list[float] = []

    for it in range(1, max_iters + 1):
        f1 = fresnel_propagation_ASM(obj, wl, dx, z1)
        I1p = np.abs(f1) ** 2

        conv = nmse_intensity(I1p, I1)
        errors.append(conv)
        print(f"Iter {it:4d} | conv={conv:.6e}")

        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping early.")
            break
        if (target_conv is not None) and (conv <= target_conv):
            print(f"Reached target convergence {target_conv:.3e} at iter {it}.")
            break

        f1 = relax_amp(f1, amp1, gamma_sen)
        b0 = fresnel_propagation_ASM(f1, wl, dx, -z1)

        obj = relax_amp(b0, amp_obj, gamma_obj)
        obj *= mask

        if fix_piston:
            phi0 = circular_mean_phase(obj, mask_bool)
            obj *= np.exp(-1j * phi0)

    f1_final = fresnel_propagation_ASM(obj, wl, dx, z1)
    return obj, f1_final, np.array(errors, dtype=np.float64)


def run_two_plane(
    I_obj: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    mask: np.ndarray,
    wl: float,
    dx: float,
    z1: float,
    z2: float,
    init: str,
    max_iters: int,
    gamma_obj: float,
    gamma_sen: float,
    target_conv: float | None,
    allow_esc: bool,
    fix_piston: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amp_obj = np.sqrt(np.maximum(I_obj, 0.0))
    amp1 = np.sqrt(np.maximum(I1, 0.0))
    amp2 = np.sqrt(np.maximum(I2, 0.0))

    ny, nx = I_obj.shape
    if init == "random":
        rng = np.random.default_rng(1234)
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    else:
        phase0 = np.zeros((ny, nx), dtype=np.float64)

    obj = amp_obj * np.exp(1j * phase0)
    obj *= mask

    dz = z2 - z1
    mask_bool = mask > 0.5
    errors: list[float] = []

    for it in range(1, max_iters + 1):
        # obj -> z1
        f1 = fresnel_propagation_ASM(obj, wl, dx, z1)
        I1p = np.abs(f1) ** 2
        f1 = relax_amp(f1, amp1, gamma_sen)

        # z1 -> z2
        f2 = fresnel_propagation_ASM(f1, wl, dx, dz)
        I2p = np.abs(f2) ** 2
        f2 = relax_amp(f2, amp2, gamma_sen)

        # back z2 -> z1
        b1 = fresnel_propagation_ASM(f2, wl, dx, -dz)
        b1 = relax_amp(b1, amp1, gamma_sen)

        # back z1 -> obj
        b0 = fresnel_propagation_ASM(b1, wl, dx, -z1)

        obj = relax_amp(b0, amp_obj, gamma_obj)
        obj *= mask

        e1 = nmse_intensity(I1p, I1)
        e2 = nmse_intensity(I2p, I2)
        conv = 0.5 * (e1 + e2)
        errors.append(conv)

        print(f"Iter {it:4d} | conv={conv:.6e} (e1={e1:.3e}, e2={e2:.3e})")

        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping early.")
            break
        if (target_conv is not None) and (conv <= target_conv):
            print(f"Reached target convergence {target_conv:.3e} at iter {it}.")
            break

        if fix_piston:
            phi0 = circular_mean_phase(obj, mask_bool)
            obj *= np.exp(-1j * phi0)

    f1_final = fresnel_propagation_ASM(obj, wl, dx, z1)
    f2_final = fresnel_propagation_ASM(obj, wl, dx, z2)
    return obj, f1_final, f2_final, np.array(errors, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser("Fresnel phase retrieval (Misell multi-plane GS, Pylance-safe)")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--max_iters", type=int, default=600)
    ap.add_argument("--target_conv", type=float, default=None)
    ap.add_argument("--init", choices=["zero", "random"], default="zero")
    ap.add_argument("--gamma_obj", type=float, default=1.0)
    ap.add_argument("--gamma_sen", type=float, default=1.0)
    ap.add_argument("--no_esc", action="store_true")
    ap.add_argument("--no_fix_piston", action="store_true")
    ap.add_argument("--out_prefix", type=str, default="run_")
    args = ap.parse_args()

    data = np.load(args.dataset)

    I_obj = data["I_obj"]
    I1 = data["I_sensor1"]
    obj_gt = data["object_complex"]
    phase_target = data["phase_target"]
    mask = data["support_mask"].astype(np.float64)
    mask_bool = mask > 0.5

    wl = float(data["wavelength"])
    dx = float(data["pixel_size"])
    z1 = float(data["z1"])

    has2 = ("I_sensor2" in data.files) and ("z2" in data.files)

    # -------- sanity check：确认数据集自洽（如果不是≈0，说明你混了文件/参数）--------
    f1_gt = fresnel_propagation_ASM(obj_gt, wl, dx, z1)
    e1_gt = nmse_intensity(np.abs(f1_gt) ** 2, I1)
    if has2:
        I2_check = data["I_sensor2"]
        z2_check = float(data["z2"])
        f2_gt = fresnel_propagation_ASM(obj_gt, wl, dx, z2_check)
        e2_gt = nmse_intensity(np.abs(f2_gt) ** 2, I2_check)
        print(f"[Sanity] forward NMSE vs stored: z1={e1_gt:.3e}, z2={e2_gt:.3e}")
    else:
        print(f"[Sanity] forward NMSE vs stored: z1={e1_gt:.3e}")
    # ---------------------------------------------------------------------------

    allow_esc = not args.no_esc
    fix_piston = not args.no_fix_piston

    if has2:
        I2 = data["I_sensor2"]
        z2 = float(data["z2"])
        obj_rec, f1_rec, f2_rec, errors = run_two_plane(
            I_obj, I1, I2, mask, wl, dx, z1, z2,
            init=args.init,
            max_iters=args.max_iters,
            gamma_obj=args.gamma_obj,
            gamma_sen=args.gamma_sen,
            target_conv=args.target_conv,
            allow_esc=allow_esc,
            fix_piston=fix_piston,
        )
        np.save(args.out_prefix + "recovered_sensor2_complex.npy", f2_rec)
    else:
        obj_rec, f1_rec, errors = run_single_plane(
            I_obj, I1, mask, wl, dx, z1,
            init=args.init,
            max_iters=args.max_iters,
            gamma_obj=args.gamma_obj,
            gamma_sen=args.gamma_sen,
            target_conv=args.target_conv,
            allow_esc=allow_esc,
            fix_piston=fix_piston,
        )

    np.save(args.out_prefix + "recovered_object_complex.npy", obj_rec)
    np.save(args.out_prefix + "recovered_sensor1_complex.npy", f1_rec)
    np.savetxt(args.out_prefix + "error_history.txt", errors)

    nmse_obj = nmse_complex(obj_gt, obj_rec, mask=mask_bool)
    print(f"Final NMSE (object complex, within mask) = {nmse_obj:.6e}")
    if errors.size:
        print(f"Final convergence = {errors[-1]:.6e}")

    # 显示：对齐一个全局相位（解决“偏置”展示问题）
    obj_plot, phi_align = align_global_phase(obj_rec, obj_gt, mask_bool)
    print(f"Visualization: applied global phase alignment phi0={phi_align:.6f} rad")

    phase_rec = np.angle(obj_plot).copy()
    phase_tgt = phase_target.copy()
    phase_rec[~mask_bool] = np.nan
    phase_tgt[~mask_bool] = np.nan

    vmin, vmax = -3.15, 3.15
    cmap = plt.get_cmap("twilight").copy()
    cmap.set_bad(color="black")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(phase_rec, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Recovered object phase [rad]")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.xticks([]); plt.yticks([])

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(phase_tgt, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Target object phase [rad]")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.savefig(args.out_prefix + "phase_comparison.png", dpi=150)
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.plot(errors, marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Convergence (NMSE)")
    plt.yscale("log")
    plt.title("GS convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_prefix + "error_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
