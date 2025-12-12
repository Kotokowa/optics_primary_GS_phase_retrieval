import numpy as np
import matplotlib.pyplot as plt
import argparse


def fresnel_propagation_ASM(field, wavelength, pixel_size, z, bandlimit_evanescent=True):
    k = 2 * np.pi / wavelength
    ny, nx = field.shape
    dx = dy = pixel_size

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    fsq = FX**2 + FY**2
    arg = 1.0 - (wavelength**2) * fsq

    if bandlimit_evanescent:
        arg = np.maximum(arg, 0.0)
        H = np.exp(1j * k * z * np.sqrt(arg))
    else:
        H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    return np.fft.ifft2(np.fft.fft2(field) * H)


def check_escape_pressed():
    try:
        import msvcrt
    except ImportError:
        return False
    if msvcrt.kbhit():
        return msvcrt.getch() == b"\x1b"
    return False


def nmse_complex(gt, rec, mask=None):
    gt = gt.astype(np.complex128)
    rec = rec.astype(np.complex128)
    if mask is not None:
        gt = gt[mask]
        rec = rec[mask]
    alpha = np.vdot(rec, gt) / (np.vdot(rec, rec) + 1e-30)
    diff = gt - alpha * rec
    return np.mean(np.abs(diff) ** 2) / (np.mean(np.abs(gt) ** 2) + 1e-30)


def circular_mean_phase(field, mask_bool):
    z = field[mask_bool]
    s = np.sum(z / (np.abs(z) + 1e-30))
    return np.angle(s + 1e-30)


def unwrap2d(ph):
    ph = np.unwrap(ph, axis=1)
    ph = np.unwrap(ph, axis=0)
    return ph


def align_piston_tilt(rec, ref, mask_bool, weight=None):
    """
    用于展示/对比：拟合 phase(rec*conj(ref)) ≈ a*x + b*y + c
    """
    ny, nx = rec.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx - nx // 2).astype(np.float64)
    y = (yy - ny // 2).astype(np.float64)

    ratio = rec * np.conj(ref)
    ph = unwrap2d(np.angle(ratio))

    m = mask_bool
    A = np.stack([x[m], y[m], np.ones(np.count_nonzero(m))], axis=1)
    b = ph[m]

    if weight is None:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        w = weight[m].astype(np.float64)
        W = np.sqrt(w / (np.mean(w) + 1e-30))
        Aw = A * W[:, None]
        bw = b * W
        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    a, b1, c = coef
    corr = np.exp(-1j * (a * x + b1 * y + c))
    return rec * corr, (a, b1, c)


def relax_to_amplitude(field, amp_target, gamma):
    """relaxation：gamma=1 等价于硬替换幅度；gamma<1 更稳。"""
    if gamma >= 1.0:
        return amp_target * np.exp(1j * np.angle(field))
    return (1 - gamma) * field + gamma * (amp_target * np.exp(1j * np.angle(field)))


def gs_two_plane(
    I_obj, I1, I2, wavelength, pixel_size, z1, z2, mask,
    max_iters=500, init="zero", gamma_obj=1.0, gamma_sen=1.0,
    enforce_support=True, fix_piston=True, allow_esc=True, target_conv=None,
    bandlimit_evanescent=True,
):
    amp_obj = np.sqrt(np.maximum(I_obj, 0.0))
    amp1 = np.sqrt(np.maximum(I1, 0.0))
    amp2 = np.sqrt(np.maximum(I2, 0.0))
    ny, nx = I_obj.shape
    mask_bool = mask > 0.5

    if init == "random":
        rng = np.random.default_rng()
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    else:
        phase0 = np.zeros((ny, nx), dtype=np.float64)

    obj = amp_obj * np.exp(1j * phase0)
    if enforce_support:
        obj *= mask

    dz12 = z2 - z1
    errors = []

    for it in range(1, max_iters + 1):
        # obj -> plane1
        f1 = fresnel_propagation_ASM(obj, wavelength, pixel_size, z1, bandlimit_evanescent)
        I1p = np.abs(f1) ** 2

        # plane1 amplitude constraint (relax)
        f1 = relax_to_amplitude(f1, amp1, gamma_sen)

        # plane1 -> plane2
        f2 = fresnel_propagation_ASM(f1, wavelength, pixel_size, dz12, bandlimit_evanescent)
        I2p = np.abs(f2) ** 2

        # plane2 amplitude constraint
        f2 = relax_to_amplitude(f2, amp2, gamma_sen)

        # errors (平均两平面)
        e1 = np.mean((I1p - I1) ** 2) / (np.mean(I1 ** 2) + 1e-30)
        e2 = np.mean((I2p - I2) ** 2) / (np.mean(I2 ** 2) + 1e-30)
        err = 0.5 * (e1 + e2)
        errors.append(err)
        print(f"Iter {it:4d} | convergence = {err:.6e} (e1={e1:.3e}, e2={e2:.3e})")

        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping early.")
            break
        if (target_conv is not None) and (err <= target_conv):
            print(f"Reached target convergence {target_conv:.3e} at iter {it}.")
            break

        # back: plane2 -> plane1
        b1 = fresnel_propagation_ASM(f2, wavelength, pixel_size, -dz12, bandlimit_evanescent)
        # (可选)再加一次 plane1 约束通常更稳
        b1 = relax_to_amplitude(b1, amp1, gamma_sen)

        # back: plane1 -> obj
        b0 = fresnel_propagation_ASM(b1, wavelength, pixel_size, -z1, bandlimit_evanescent)

        # obj amplitude constraint (relax)
        obj = relax_to_amplitude(b0, amp_obj, gamma_obj)

        if enforce_support:
            obj *= mask

        if fix_piston:
            phi0 = circular_mean_phase(obj, mask_bool)
            obj *= np.exp(-1j * phi0)

    # final forward fields for output
    f1 = fresnel_propagation_ASM(obj, wavelength, pixel_size, z1, bandlimit_evanescent)
    f2 = fresnel_propagation_ASM(obj, wavelength, pixel_size, z2, bandlimit_evanescent)
    return obj, f1, f2, np.array(errors)


def gs_single_plane(
    I_obj, I1, wavelength, pixel_size, z1, mask,
    max_iters=500, init="zero", gamma_obj=1.0, gamma_sen=1.0,
    enforce_support=True, fix_piston=True, allow_esc=True, target_conv=None,
    bandlimit_evanescent=True,
):
    amp_obj = np.sqrt(np.maximum(I_obj, 0.0))
    amp1 = np.sqrt(np.maximum(I1, 0.0))
    ny, nx = I_obj.shape
    mask_bool = mask > 0.5

    if init == "random":
        rng = np.random.default_rng()
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    else:
        phase0 = np.zeros((ny, nx), dtype=np.float64)

    obj = amp_obj * np.exp(1j * phase0)
    if enforce_support:
        obj *= mask

    errors = []

    for it in range(1, max_iters + 1):
        f1 = fresnel_propagation_ASM(obj, wavelength, pixel_size, z1, bandlimit_evanescent)
        I1p = np.abs(f1) ** 2

        e1 = np.mean((I1p - I1) ** 2) / (np.mean(I1 ** 2) + 1e-30)
        errors.append(e1)
        print(f"Iter {it:4d} | convergence = {e1:.6e}")

        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping early.")
            break
        if (target_conv is not None) and (e1 <= target_conv):
            print(f"Reached target convergence {target_conv:.3e} at iter {it}.")
            break

        f1 = relax_to_amplitude(f1, amp1, gamma_sen)
        b0 = fresnel_propagation_ASM(f1, wavelength, pixel_size, -z1, bandlimit_evanescent)
        obj = relax_to_amplitude(b0, amp_obj, gamma_obj)

        if enforce_support:
            obj *= mask

        if fix_piston:
            phi0 = circular_mean_phase(obj, mask_bool)
            obj *= np.exp(-1j * phi0)

    f1 = fresnel_propagation_ASM(obj, wavelength, pixel_size, z1, bandlimit_evanescent)
    return obj, f1, np.array(errors)


def main():
    parser = argparse.ArgumentParser(description="Program 3: Fresnel GS (single or two-plane)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, default="run_")
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--target_conv", type=float, default=None)
    parser.add_argument("--init", choices=["zero", "random"], default="zero")
    parser.add_argument("--restarts", type=int, default=1, help="for random init, try multiple and keep best")
    parser.add_argument("--gamma_obj", type=float, default=1.0)
    parser.add_argument("--gamma_sen", type=float, default=1.0)
    parser.add_argument("--no_support", action="store_true")
    parser.add_argument("--no_fix_piston", action="store_true")
    parser.add_argument("--no_esc", action="store_true")
    parser.add_argument("--keep_evanescent", action="store_true")
    args = parser.parse_args()
    if args.restarts < 1:
        raise ValueError("--restarts must be >= 1")

    data = np.load(args.dataset)
    I_obj = data["I_obj"]
    I1 = data["I_sensor1"]
    obj_gt = data["object_complex"]
    phase_target = data["phase_target"]

    wavelength = float(data["wavelength"])
    pixel_size = float(data["pixel_size"])
    z1 = float(data["z1"])
    mask = data["support_mask"].astype(np.float64) if "support_mask" in data.files else (I_obj > 1e-12*np.max(I_obj)).astype(np.float64)
    mask_bool = mask > 0.5

    two_plane = "I_sensor2" in data.files
    I2 = data["I_sensor2"] if two_plane else None
    z2 = float(data["z2"]) if ("z2" in data.files) else None

    bandlimit = not args.keep_evanescent
    enforce_support = not args.no_support
    fix_piston = not args.no_fix_piston
    allow_esc = not args.no_esc

    best = None
    best_err = np.inf

    for r in range(args.restarts):
        init_mode = args.init
        if args.restarts > 1 and args.init == "zero":
            init_mode = "random"  # 多次重启时，默认用随机更有意义

        if two_plane:
            obj_rec, f1_rec, f2_rec, errors = gs_two_plane(
                I_obj, I1, I2, wavelength, pixel_size, z1, z2, mask,
                max_iters=args.max_iters,
                init=init_mode,
                gamma_obj=args.gamma_obj,
                gamma_sen=args.gamma_sen,
                enforce_support=enforce_support,
                fix_piston=fix_piston,
                allow_esc=allow_esc,
                target_conv=args.target_conv,
                bandlimit_evanescent=bandlimit,
            )
            final_err = float(errors[-1]) if errors.size else np.inf
        else:
            obj_rec, f1_rec, errors = gs_single_plane(
                I_obj, I1, wavelength, pixel_size, z1, mask,
                max_iters=args.max_iters,
                init=init_mode,
                gamma_obj=args.gamma_obj,
                gamma_sen=args.gamma_sen,
                enforce_support=enforce_support,
                fix_piston=fix_piston,
                allow_esc=allow_esc,
                target_conv=args.target_conv,
                bandlimit_evanescent=bandlimit,
            )
            f2_rec = None
            final_err = float(errors[-1]) if errors.size else np.inf

        if (best is None) or (final_err < best_err):
            best_err = final_err
            best = (obj_rec, f1_rec, f2_rec, errors)

        if args.restarts > 1:
            print(f"[Restart {r+1}/{args.restarts}] final convergence = {final_err:.6e}")

    if best is None:
        raise RuntimeError("No valid run produced results. Check --restarts and whether the GS loop executed.")

    obj_rec, f1_rec, f2_rec, errors = best

    # 保存
    np.save(args.out_prefix + "recovered_object_complex.npy", obj_rec)
    np.save(args.out_prefix + "recovered_sensor1_complex.npy", f1_rec)
    if f2_rec is not None:
        np.save(args.out_prefix + "recovered_sensor2_complex.npy", f2_rec)
    np.savetxt(args.out_prefix + "error_history.txt", errors)

    # 评价 NMSE（支持域内）
    nmse_obj = nmse_complex(obj_gt, obj_rec, mask=mask_bool)
    print(f"Final NMSE (object complex, within mask) = {nmse_obj:.6e}")
    print(f"Final convergence = {errors[-1]:.6e}")

    # === 解释你说的“偏置”：强度本身无法决定 piston/tilt，这里只用于展示对齐 ===
    amp_w = np.sqrt(np.maximum(I_obj, 0.0))
    obj_aligned, (a, b, c) = align_piston_tilt(obj_rec, obj_gt, mask_bool, weight=amp_w)
    print(f"Visualization alignment (piston+tilt): a={a:.3e}, b={b:.3e}, c={c:.3e}")

    # 画相位（mask 外 NaN）
    phase_rec = np.angle(obj_aligned).copy()
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

    # 误差曲线
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
