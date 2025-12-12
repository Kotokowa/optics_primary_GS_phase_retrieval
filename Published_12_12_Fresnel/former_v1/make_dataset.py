import numpy as np
import matplotlib.pyplot as plt


def fresnel_propagation_ASM(field, wavelength, pixel_size, z):
    """
    基于角谱法的 Fresnel 传播（标量近轴模型）

    field:      2D complex ndarray, 输入场 (z=0)
    wavelength: 波长 (m)
    pixel_size: 像素尺寸 (m)，x,y 相同
    z:          传播距离 (m)，可正可负
    """
    k = 2 * np.pi / wavelength
    ny, nx = field.shape
    dx = dy = pixel_size

    fx = np.fft.fftfreq(nx, d=dx)  # cycles/m
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    fsq = FX**2 + FY**2
    arg = 1.0 - (wavelength**2) * fsq

    # 允许复数开方以包含倏逝波分量
    H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    F_field = np.fft.fft2(field)
    out_field = np.fft.ifft2(F_field * H)
    return out_field


def main():
    # -------- 用户参数区域（根据需要修改） --------
    object_file = "object_complex.npy"  # 程序1生成的物面复振幅
    wavelength = 532e-9                 # 波长 (m)
    pixel_size = 8e-6                   # 像素尺寸 (m)
    z = 0.10                            # 传播距离 z (m)
    out_dataset = "gs_dataset.npz"      # 输出数据集文件名

    # xyz 三方向位移，这里只真正用到 z，其余做元数据占位
    shift_x = 0.0
    shift_y = 0.0
    shift_z = z
    # -------------------------------------------

    # 读取物面复数场
    object_complex = np.load(object_file)
    intensity_obj = np.abs(object_complex)**2
    phase_obj = np.angle(object_complex)

    # Fresnel 传播到传感器面
    sensor_complex = fresnel_propagation_ASM(object_complex, wavelength, pixel_size, z)
    intensity_sensor = np.abs(sensor_complex)**2

    # 构建并保存数据集
    np.savez(
        out_dataset,
        I_obj=intensity_obj,              # 物面强度（第一个强度信息）
        I_sensor=intensity_sensor,        # 传感器面强度（第二个强度信息）
        phase_target=phase_obj,           # 目标相位信息（物面）
        intensity_target=intensity_obj,   # 目标强度信息（这里等于物面强度）
        object_complex=object_complex,    # 目标物面复振幅（用于评估 NMSE）
        sensor_complex=sensor_complex,    # 对应传感器面复振幅
        wavelength=wavelength,
        pixel_size=pixel_size,
        shift_xyz=np.array([shift_x, shift_y, shift_z]),
        z=z
    )

    print(f"Saved dataset to: {out_dataset}")
    print("  keys:")
    print("   - I_obj (object-plane intensity)")
    print("   - I_sensor (sensor-plane intensity)")
    print("   - phase_target (object-plane phase)")
    print("   - intensity_target (object-plane intensity, alias)")
    print("   - object_complex (ground truth at object)")
    print("   - sensor_complex (ground truth at sensor)")
    print("   - wavelength, pixel_size, shift_xyz, z")

    # 简单可视化一下传感器面强度
    plt.figure(figsize=(5, 4))
    plt.imshow(intensity_sensor, cmap="gray")
    plt.title("Sensor intensity")
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
