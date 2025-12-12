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


def main():
    parser = argparse.ArgumentParser(description="Program 2: build dataset with 1 or 2 sensor planes")
    parser.add_argument("--obj_prefix", type=str, required=True)
    parser.add_argument("--out", type=str, default="gs_dataset.npz")
    parser.add_argument("--wavelength", type=float, default=532e-9)
    parser.add_argument("--pixel_size", type=float, default=8e-6)
    parser.add_argument("--z1", type=float, default=0.10)
    parser.add_argument("--z2", type=float, default=None, help="second plane z (meters). if omitted -> single-plane")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--keep_evanescent", action="store_true")
    args = parser.parse_args()

    obj = np.load(args.obj_prefix + "complex.npy")
    mask = np.load(args.obj_prefix + "mask.npy").astype(np.float64)

    I_obj = np.abs(obj) ** 2
    phase_target = np.angle(obj)

    bandlimit = not args.keep_evanescent

    sen1 = fresnel_propagation_ASM(obj, args.wavelength, args.pixel_size, args.z1, bandlimit)
    I1 = np.abs(sen1) ** 2

    payload = dict(
        I_obj=I_obj,
        phase_target=phase_target,
        intensity_target=I_obj,
        object_complex=obj,
        support_mask=mask,
        wavelength=float(args.wavelength),
        pixel_size=float(args.pixel_size),
        z1=float(args.z1),
        obj_prefix=args.obj_prefix,
        I_sensor1=I1,
    )

    if args.z2 is not None:
        sen2 = fresnel_propagation_ASM(obj, args.wavelength, args.pixel_size, args.z2, bandlimit)
        I2 = np.abs(sen2) ** 2
        payload["z2"] = float(args.z2)
        payload["I_sensor2"] = I2

    np.savez(args.out, **payload)
    print(f"Saved dataset: {args.out}")
    print(f"  planes: z1={args.z1:.4f}" + (f", z2={args.z2:.4f}" if args.z2 is not None else ""))

    if args.show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(I1, cmap="gray")
        plt.title("Sensor intensity @ z1")
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        if args.z2 is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(I2, cmap="gray")
            plt.title("Sensor intensity @ z2")
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
