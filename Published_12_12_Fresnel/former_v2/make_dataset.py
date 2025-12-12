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

    # 为了 GS 回传稳定，建议抑制倏逝波（尤其 z 取负时会指数放大）
    if bandlimit_evanescent:
        arg = np.maximum(arg, 0.0)
        H = np.exp(1j * k * z * np.sqrt(arg))
    else:
        H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    out_field = np.fft.ifft2(np.fft.fft2(field) * H)
    return out_field


def main():
    parser = argparse.ArgumentParser(description="Program 2: build GS dataset (.npz)")
    parser.add_argument("--obj_prefix", type=str, required=True, help="prefix from Program1, e.g. snellen_")
    parser.add_argument("--out", type=str, default="gs_dataset.npz")
    parser.add_argument("--wavelength", type=float, default=532e-9, help="meters")
    parser.add_argument("--pixel_size", type=float, default=8e-6, help="meters")
    parser.add_argument("--z", type=float, default=0.10, help="meters")
    parser.add_argument("--show", action="store_true", help="visualize sensor intensity")
    args = parser.parse_args()

    obj_complex_file = args.obj_prefix + "complex.npy"
    mask_file = args.obj_prefix + "mask.npy"

    object_complex = np.load(obj_complex_file)
    support_mask = np.load(mask_file)

    intensity_obj = np.abs(object_complex)**2
    phase_obj = np.angle(object_complex)

    sensor_complex = fresnel_propagation_ASM(object_complex, args.wavelength, args.pixel_size, args.z)
    intensity_sensor = np.abs(sensor_complex)**2

    shift_xyz = np.array([0.0, 0.0, args.z], dtype=np.float64)

    np.savez(
        args.out,
        I_obj=intensity_obj,
        I_sensor=intensity_sensor,
        phase_target=phase_obj,
        intensity_target=intensity_obj,
        object_complex=object_complex,
        sensor_complex=sensor_complex,
        support_mask=support_mask,
        wavelength=float(args.wavelength),
        pixel_size=float(args.pixel_size),
        shift_xyz=shift_xyz,
        z=float(args.z),
        obj_prefix=args.obj_prefix,
    )

    print(f"Saved dataset to: {args.out}")
    print(f"  loaded: {obj_complex_file}, {mask_file}")
    print(f"  wavelength={args.wavelength:.3e} m, pixel_size={args.pixel_size:.3e} m, z={args.z:.3e} m")

    if args.show:
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
