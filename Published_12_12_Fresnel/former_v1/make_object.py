import numpy as np
import matplotlib.pyplot as plt


def create_xy_grid(N, pixel_size):
    """
    生成 N x N 的空间坐标 X, Y（单位：米）
    """
    x = (np.arange(N) - N // 2) * pixel_size
    X, Y = np.meshgrid(x, x)
    return X, Y


def make_example_intensity(X, Y, kind="gaussian"):
    """
    生成一个示例物面强度分布（>=0）
    返回的是「强度」而不是振幅。
    你可以直接改这里做自己的模式。
    """
    R2 = X**2 + Y**2
    if kind == "gaussian":
        sigma = 0.4 * np.max(np.abs(X))
        amp = np.exp(-R2 / (2 * sigma**2))
        intensity = amp**2
    elif kind == "circle":
        r0 = 0.5 * np.max(np.abs(X))
        mask = (R2 <= r0**2).astype(float)
        intensity = mask
    elif kind == "ring":
        r0 = 0.2 * np.max(np.abs(X))
        r1 = 0.4 * np.max(np.abs(X))
        R = np.sqrt(R2)
        mask = ((R >= r0) & (R <= r1)).astype(float)
        intensity = mask
    else:
        # 默认：圆形孔径
        r0 = 0.5 * np.max(np.abs(X))
        mask = (R2 <= r0**2).astype(float)
        intensity = mask
    return intensity


def make_example_phase(X, Y, kind="quadratic"):
    """
    生成一个示例物面相位分布（单位：弧度）
    """
    R2 = X**2 + Y**2
    if kind == "quadratic":
        r0 = np.max(np.abs(X)) + 1e-12
        phase = 2.5 * np.pi * R2 / (r0**2)  # 大致在 [-2.5pi, 2.5pi]
    elif kind == "tilt_x":
        phase = np.pi * X / (np.max(np.abs(X)) + 1e-12)
    elif kind == "tilt_y":
        phase = np.pi * Y / (np.max(np.abs(Y)) + 1e-12)
    elif kind == "astig":
        r0 = np.max(np.abs(X)) + 1e-12
        phase = np.pi * (X**2 - Y**2) / (r0**2)
    else:
        phase = np.zeros_like(X)
    return phase


def main():
    # -------- 用户参数区域（根据需要修改） --------
    N = 256                      # 物面采样点数（N x N）
    pixel_size = 8e-6            # 空间采样间隔（米）
    intensity_kind = "circle"  # "gaussian", "circle", "ring", ...
    phase_kind = "quadratic"     # "quadratic", "tilt_x", "tilt_y", "astig", ...
    out_prefix = "object_"       # 输出文件名前缀
    # -------------------------------------------

    X, Y = create_xy_grid(N, pixel_size)
    intensity = make_example_intensity(X, Y, kind=intensity_kind)
    phase = make_example_phase(X, Y, kind=phase_kind)

    # 物面复数场： amplitude * exp(i * phase)
    amplitude = np.sqrt(intensity)
    object_complex = amplitude * np.exp(1j * phase)

    # 保存为 .npy 文件
    np.save(out_prefix + "complex.npy", object_complex)
    np.save(out_prefix + "intensity.npy", intensity)
    np.save(out_prefix + "phase.npy", phase)

    print("Saved:")
    print("  ", out_prefix + "complex.npy")
    print("  ", out_prefix + "intensity.npy")
    print("  ", out_prefix + "phase.npy")

    # 可视化物面强度和相位
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(intensity, cmap="gray")
    axs[0].set_title("Object intensity")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Object phase [rad]")
    plt.colorbar(im1, ax=axs[1])

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
