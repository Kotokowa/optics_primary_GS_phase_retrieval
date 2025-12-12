import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def fresnel_propagation_ASM(field, wavelength, pixel_size, z):
    """
    基于角谱法的 Fresnel 传播。

    field:      2D complex ndarray, 输入场 (z=0)
    wavelength: 波长 (m)
    pixel_size: 像素尺寸 (m)
    z:          传播距离 (m)，可正可负
    """
    k = 2 * np.pi / wavelength
    ny, nx = field.shape
    dx = dy = pixel_size

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    fsq = FX**2 + FY**2
    arg = 1.0 - (wavelength**2) * fsq

    H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    F_field = np.fft.fft2(field)
    out_field = np.fft.ifft2(F_field * H)
    return out_field


def check_escape_pressed():
    """
    检测是否按下 ESC（仅 Windows 有效）。
    其它系统上总是返回 False。
    """
    try:
        import msvcrt
    except ImportError:
        return False

    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch == b'\x1b':  # ESC
            return True
    return False


def nmse_complex(gt, rec):
    """
    计算两个复场之间的归一化 MSE（允许一个全局复数标量）
    """
    gt = gt.astype(np.complex128)
    rec = rec.astype(np.complex128)
    num = np.vdot(rec, gt)        # rec^H * gt
    den = np.vdot(rec, rec) + 1e-30
    alpha = num / den
    diff = gt - alpha * rec
    n = np.mean(np.abs(diff)**2)
    d = np.mean(np.abs(gt)**2) + 1e-30
    return n / d


def align_global_phase(rec, ref, mask=None):
    """
    用一个全局相位因子把 rec 对齐到 ref：
      rec_aligned = rec * exp(i*phi0)
    mask: True 表示参与对齐的区域（建议用支持域）
    返回：rec_aligned, phi0
    """
    if mask is None:
        mask = np.ones(rec.shape, dtype=bool)

    rec_m = rec[mask]
    ref_m = ref[mask]

    alpha = np.vdot(rec_m, ref_m) / (np.vdot(rec_m, rec_m) + 1e-30)
    phase_factor = alpha / (np.abs(alpha) + 1e-30)   # 只取相位，不改幅度
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
    allow_esc=True,
):
    """
    Fresnel 基的 Gerchberg-Saxton 算法：

      - 输入：
          I_obj:    物面强度
          I_sensor: 传感器面强度
      - 传播算子：角谱法 Fresnel 传播

      - 输出：
          object_complex (恢复物面复场)
          sensor_complex (由恢复物面场前向传播得到)
          errors: 每次迭代的 NMSE（传感器面强度）
          nmse_obj: 恢复物面复场相对真值的 NMSE（若提供真值）
    """
    amp_obj = np.sqrt(I_obj)
    ny, nx = I_obj.shape

    # 初始物面随机相位
    if init_mode == "random":
        rng = np.random.default_rng()
        phase0 = rng.random((ny, nx)) * 2 * np.pi
    elif init_mode == "zero":
        phase0 = np.zeros_like(I_obj)
    else:
        raise ValueError("unknown init_mode")

    obj_field = amp_obj * np.exp(1j * phase0)

    errors = []

    for it in range(1, max_iters + 1):
        # 1) 物面 -> 传感器面
        sensor_field = fresnel_propagation_ASM(obj_field, wavelength, pixel_size, z)
        I_sensor_pred = np.abs(sensor_field)**2

        # 2) 计算传感器面强度 NMSE 作为收敛指标
        num = np.mean((I_sensor_pred - I_sensor)**2)
        den = np.mean(I_sensor**2) + 1e-30
        err = num / den
        errors.append(err)

        print(f"Iter {it:4d} | convergence (NMSE on sensor intensity) = {err:.6e}")

        # ESC 提前中止
        if allow_esc and check_escape_pressed():
            print("ESC detected, stopping iterations early.")
            break

        # 收敛判据
        if (target_convergence is not None) and (err <= target_convergence):
            print(f"Reached target convergence {target_convergence:.3e} at iter {it}.")
            break

        # 3) 传感器面施加强度约束
        sensor_phase = np.angle(sensor_field)
        sensor_field = np.sqrt(I_sensor) * np.exp(1j * sensor_phase)

        # 4) 回传到物面
        obj_field_back = fresnel_propagation_ASM(sensor_field, wavelength, pixel_size, -z)

        # 5) 物面施加强度约束（使用 I_obj）
        obj_phase = np.angle(obj_field_back)
        obj_field = amp_obj * np.exp(1j * obj_phase)

        # ===== 相位基准固定（可选）=====
        # 方案1：中心像素相位设为0
        ph_ref = np.angle(obj_field[ny//2, nx//2])
        obj_field *= np.exp(-1j * ph_ref)

    # 用最终物面场再前向传播一次，得到对应传感器面场
    sensor_field_final = fresnel_propagation_ASM(obj_field, wavelength, pixel_size, z)

    nmse_obj = None
    if obj_complex_gt is not None:
        nmse_obj = nmse_complex(obj_complex_gt, obj_field)
        print(f"Final NMSE (recovered object complex vs. ground truth): {nmse_obj:.6e}")

    return obj_field, sensor_field_final, np.array(errors), nmse_obj


def main():
    parser = argparse.ArgumentParser(
        description="Fresnel-based Gerchberg-Saxton phase retrieval"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gs_dataset.npz",
        help="Program 2 生成的 npz 数据集 (default: gs_dataset.npz)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=200,
        help="最大迭代次数 (default: 200)",
    )
    parser.add_argument(
        "--target_conv",
        type=float,
        default=None,
        help="收敛目标（传感器面强度 NMSE），若设定则误差低于此值即停止。",
    )
    parser.add_argument(
        "--no_esc",
        action="store_true",
        help="禁用 ESC 提前终止（非 Windows 系统建议加这个）。",
    )
    parser.add_argument(
    "--init",
    type=str,
    choices=["random", "zero"],
    default="zero",
    help="GS 初值方式（random 或 zero，推荐 zero）"
    )
    args = parser.parse_args()

    data = np.load(args.dataset)

    I_obj = data["I_obj"]
    I_sensor = data["I_sensor"]
    phase_target = data["phase_target"]
    intensity_target = data["intensity_target"]
    object_complex_gt = data["object_complex"]
    wavelength = float(data["wavelength"])
    pixel_size = float(data["pixel_size"])
    z = float(data["z"])

    print("Loaded dataset:", args.dataset)
    print("  shape:", I_obj.shape)
    print(f"  wavelength = {wavelength:.3e} m")
    print(f"  pixel_size = {pixel_size:.3e} m")
    print(f"  z          = {z:.3e} m")

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
        allow_esc=not args.no_esc,
    )

    support = I_obj > (1e-12 * np.max(I_obj))  # 支持域：强度非零区域

    obj_for_plot = obj_rec
    phi0 = 0.0

    if "object_complex" in data.files:
        obj_gt = data["object_complex"]
        obj_for_plot, phi0 = align_global_phase(obj_rec, obj_gt, mask=support)
        print(f"Applied global phase alignment (phi0 = {phi0:.6f} rad) for visualization.")


    # 保存复振幅结果
    np.save("recovered_object_complex.npy", obj_rec)
    np.save("recovered_sensor_complex.npy", sensor_rec)
    np.savetxt("error_history.txt", errors)
    print("Saved:")
    print("  recovered_object_complex.npy")
    print("  recovered_sensor_complex.npy")
    print("  error_history.txt")

    # ================== 相位可视化（你重点关心的部分） ==================
    phase_rec = np.angle(obj_for_plot).copy()
    phase_tgt = phase_target.copy()

    # mask 支持域外的相位（那里相位没有意义）
    phase_rec[~support] = np.nan
    phase_tgt[~support] = np.nan

    vmin, vmax = -3.15, 3.15
    cmap = plt.get_cmap("twilight").copy()
    cmap.set_bad(color="black")   # NaN 显示为黑色（也可以改成透明）


    plt.figure(figsize=(10, 4))

    # 左：恢复的物面相位
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(phase_rec, cmap=cmap, vmin=vmin, vmax=vmax) #cmap还可选"twilight"
    plt.title("Recovered object phase [rad]")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])

    # 右：目标物面相位
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(phase_target, cmap=cmap, vmin=vmin, vmax=vmax) #cmap还可选"twilight"
    plt.title("Target object phase [rad]")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig("phase_comparison.png", dpi=150)
    plt.show()   # 新增：直接显示出来
    print("Saved phase comparison figure: phase_comparison.png")

    # （可选）额外显示一下恢复的传感器面相位
    phase_sensor_rec = np.angle(sensor_rec)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(phase_sensor_rec, cmap="twilight", vmin=vmin, vmax=vmax)
    plt.title("Recovered sensor phase [rad]")
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("sensor_phase_recovered.png", dpi=150)
    plt.show()
    print("Saved sensor phase figure: sensor_phase_recovered.png")

    # ================== 误差曲线 ==================
    plt.figure(figsize=(5, 4))
    plt.plot(errors, marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("NMSE on sensor intensity")
    plt.yscale("log")
    plt.title("GS convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_curve.png", dpi=150)
    plt.show()
    print("Saved error curve figure: error_curve.png")

    # 打印最终结果
    if nmse_obj is not None:
        print(f"Final NMSE (object complex) = {nmse_obj:.6e}")
    if errors.size > 0:
        print(f"Final convergence (sensor intensity NMSE) = {errors[-1]:.6e}")



if __name__ == "__main__":
    main()
