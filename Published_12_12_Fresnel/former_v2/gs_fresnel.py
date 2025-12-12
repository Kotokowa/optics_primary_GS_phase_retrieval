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

    # 强烈建议抑制倏逝波：避免 z<0 回传时指数放大导致不稳定
    if bandlimit_evanescent:
        arg = np.maximum(arg, 0.0)
        H = np.exp(1j * k * z * np.sqrt(arg))
    else:
        H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    out_field = np.fft.ifft2(np.fft.fft2(field) * H)
    return out_field


def check_escape_pressed():
    try:
        import msvcrt
    except ImportError:
        return False
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch == b"\x1b":
            return True
    return False


def nmse_complex(gt, rec):
    gt = gt.astype(np.complex128)
    rec = rec.astype(np.complex128)
    num = np.vdot(rec, gt)
    den = np.vdot(rec, rec) + 1e-30
    alpha = num / den
    diff = gt - alpha * rec
    n = np.mean(np.abs(diff) ** 2)
    d = np.mean(np.abs(gt) ** 2) + 1e-30
    return n / d


def align_global_phase(rec, ref, mask=None):
    if mask is None:
        mask = np.ones(rec.shape, dtype=bool)
    rec_m = rec[mask]
    ref_m = ref[mask]
    alpha = np.vdot(rec_m, ref_m) / (np.vdot(rec_m, rec_m) + 1e-30)
    phase_factor = alpha / (np.abs(alpha) + 1e-30)
    phi0 = np.angle(phase_factor)
    return rec * phase_factor, phi0


def gerchberg_saxton_fresnel(
    I_obj,
    I_sensor,
    wavelength,
    pixel_size,
    z,
    max_iters=200,
    init_mode="zero",
    target_convergence=None,
    obj_complex_gt=None,
    support_mask=None,
    enforce_support=True,
    fix_global_phase_each_iter=True,
    allow_esc=True,
):
    amp_obj = np.sqrt(np.maximum(I_obj, 0.0))
    ny, nx = I_obj.shape

    if init_mode == "random":
        rng = np.random.default_rng()
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    elif init_mode == "zero":
        phase0 = np.zeros_like(I_obj)
    else:
        raise ValueError("unknown init_mode")

    obj_field = amp_obj * np.exp(1j * phase0)

    # 如果提供 mask，建议起始场也裁掉
    if (support_mask is not None) and enforce_support:
        obj_field *= support_mask

    errors = []

    for it in range(1, max_iters + 1):
        sensor_field = fresnel_propagation_ASM(obj_field, wavelength, pixel_size, z)
        I_sensor_pred = np.abs(sensor_field) ** 2

        num = np.mean((I_sensor_pred - I_sensor) ** 2)
        den = np.mean(I_sensor ** 2) + 1e-30
        err = num / den
        errors.append(err)

        print(f"Iter {it:4d} | convergence (NMSE on sensor intensity) = {err:.6e}")

        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping iterations early.")
            break

        if (target_convergence is not None) and (err <= target_convergence):
            print(f"Reached target convergence {target_convergence:.3e} at iter {it}.")
            break

        # 传感器面强度约束
        sensor_phase = np.angle(sensor_field)
        sensor_field = np.sqrt(np.maximum(I_sensor, 0.0)) * np.exp(1j * sensor_phase)

        # 回传
        obj_back = fresnel_propagation_ASM(sensor_field, wavelength, pixel_size, -z)

        # 物面强度约束
        obj_phase = np.angle(obj_back)
        obj_field = amp_obj * np.exp(1j * obj_phase)

        # ✅ 支持域约束（修复：support_mask 通过参数传入）
        if (support_mask is not None) and enforce_support:
            obj_field *= support_mask

        # 可选：每次迭代固定一个相位基准（避免整体漂移）
        if fix_global_phase_each_iter:
            ph_ref = np.angle(obj_field[ny // 2, nx // 2])
            obj_field *= np.exp(-1j * ph_ref)

    sensor_field_final = fresnel_propagation_ASM(obj_field, wavelength, pixel_size, z)

    nmse_obj = None
    if obj_complex_gt is not None:
        nmse_obj = nmse_complex(obj_complex_gt, obj_field)
        print(f"Final NMSE (recovered object complex vs. ground truth): {nmse_obj:.6e}")

    return obj_field, sensor_field_final, np.array(errors), nmse_obj


def main():
    parser = argparse.ArgumentParser(description="Program 3: Fresnel GS phase retrieval")
    parser.add_argument("--dataset", type=str, default="gs_dataset.npz")
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--target_conv", type=float, default=None)
    parser.add_argument("--init", choices=["random", "zero"], default="zero")
    parser.add_argument("--no_esc", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="run_", help="prefix for outputs/figures")
    parser.add_argument("--no_support_enforce", action="store_true", help="disable support constraint in iterations")
    args = parser.parse_args()

    data = np.load(args.dataset)

    I_obj = data["I_obj"]
    I_sensor = data["I_sensor"]
    phase_target = data["phase_target"]
    object_complex_gt = data["object_complex"]
    wavelength = float(data["wavelength"])
    pixel_size = float(data["pixel_size"])
    z = float(data["z"])

    support = None
    if "support_mask" in data.files:
        support = data["support_mask"].astype(np.float64)

    obj_rec, sensor_rec, errors, nmse_obj = gerchberg_saxton_fresnel(
        I_obj,
        I_sensor,
        wavelength,
        pixel_size,
        z,
        max_iters=args.max_iters,
        init_mode=args.init,
        target_convergence=args.target_conv,
        obj_complex_gt=object_complex_gt,
        support_mask=support,
        enforce_support=not args.no_support_enforce,
        allow_esc=not args.no_esc,
    )

    # ===== 输出保存（统一加前缀）=====
    np.save(args.out_prefix + "recovered_object_complex.npy", obj_rec)
    np.save(args.out_prefix + "recovered_sensor_complex.npy", sensor_rec)
    np.savetxt(args.out_prefix + "error_history.txt", errors)

    # ===== 可视化：mask 外相位用 NaN 遮掉 =====
    if support is None:
        support_bool = I_obj > (1e-12 * np.max(I_obj))
    else:
        support_bool = support > 0.5

    # 为了 PPT：把 recovered 先对齐全局相位（仅用于显示/对比）
    obj_for_plot, phi0 = align_global_phase(obj_rec, object_complex_gt, mask=support_bool)
    print(f"Applied global phase alignment (phi0 = {phi0:.6f} rad) for visualization.")

    phase_rec = np.angle(obj_for_plot).copy()
    phase_tgt = phase_target.copy()
    phase_rec[~support_bool] = np.nan
    phase_tgt[~support_bool] = np.nan

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
    plt.ylabel("NMSE on sensor intensity")
    plt.yscale("log")
    plt.title("GS convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_prefix + "error_curve.png", dpi=150)
    plt.show()

    if errors.size > 0:
        print(f"Final convergence (sensor intensity NMSE) = {errors[-1]:.6e}")


if __name__ == "__main__":
    main()
