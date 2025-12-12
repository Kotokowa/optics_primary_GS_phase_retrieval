import numpy as np
import matplotlib.pyplot as plt
import argparse


def fresnel_propagation_ASM(field, wavelength, pixel_size, z):
    k = 2 * np.pi / wavelength
    ny, nx = field.shape
    dx = dy = pixel_size

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    fsq = FX**2 + FY**2
    arg = 1.0 - (wavelength**2) * fsq
    H = np.exp(1j * k * z * np.sqrt(arg.astype(np.complex128)))

    return np.fft.ifft2(np.fft.fft2(field) * H)


def main():
    ap = argparse.ArgumentParser("Program2: build dataset for Fresnel GS (two-plane)")
    ap.add_argument("--obj_prefix", type=str, required=True, help="prefix from Program1, e.g. snellen_ or optics_")
    ap.add_argument("--out", type=str, default="gs_dataset.npz")
    ap.add_argument("--wavelength", type=float, default=532e-9)
    ap.add_argument("--pixel_size", type=float, default=8e-6)
    ap.add_argument("--z1", type=float, default=0.10)
    ap.add_argument("--z2", type=float, default=0.13)
    ap.add_argument("--single_plane", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    obj = np.load(args.obj_prefix + "complex.npy")
    mask = np.load(args.obj_prefix + "mask.npy").astype(np.float64)

    I_obj = np.abs(obj)**2
    phase_target = np.angle(obj)

    f1 = fresnel_propagation_ASM(obj, args.wavelength, args.pixel_size, args.z1)
    I1 = np.abs(f1)**2

    payload = dict(
        I_obj=I_obj,
        I_sensor1=I1,
        phase_target=phase_target,
        intensity_target=I_obj,
        object_complex=obj,
        support_mask=mask,
        wavelength=float(args.wavelength),
        pixel_size=float(args.pixel_size),
        z1=float(args.z1),
        obj_prefix=args.obj_prefix,
    )

    if not args.single_plane:
        f2 = fresnel_propagation_ASM(obj, args.wavelength, args.pixel_size, args.z2)
        I2 = np.abs(f2)**2
        payload["I_sensor2"] = I2
        payload["z2"] = float(args.z2)

    np.savez(args.out, **payload)
    print(f"Saved dataset: {args.out}")
    if not args.single_plane:
        print(f"  planes: z1={args.z1:.4f}, z2={args.z2:.4f}")
    else:
        print(f"  plane: z1={args.z1:.4f}")

    if args.show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(I1, cmap="gray")
        plt.title("Sensor intensity @ z1")
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        if not args.single_plane:
            plt.subplot(1, 2, 2)
            plt.imshow(I2, cmap="gray")
            plt.title("Sensor intensity @ z2")
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
