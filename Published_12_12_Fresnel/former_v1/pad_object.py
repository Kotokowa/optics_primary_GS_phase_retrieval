import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_object(in_complex=None, in_intensity=None, in_phase=None):
    if in_complex is not None:
        field = np.load(in_complex)
        return field

    if (in_intensity is None) or (in_phase is None):
        raise ValueError("Either --in_complex or both --in_intensity and --in_phase must be provided.")

    I = np.load(in_intensity)
    ph = np.load(in_phase)
    if I.shape != ph.shape:
        raise ValueError("Intensity and phase must have the same shape.")

    field = np.sqrt(np.maximum(I, 0.0)) * np.exp(1j * ph)
    return field


def pad_center(field, pad):
    # zero padding：外面补0
    return np.pad(field, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)


def plot_phase(phase, title, vmin=-3.15, vmax=3.15, mask=None):
    ph = phase.copy()
    if mask is not None:
        ph[~mask] = np.nan

    cmap = plt.get_cmap("twilight").copy()
    cmap.set_bad(color="black")

    plt.figure(figsize=(5, 4))
    im = plt.imshow(ph, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Zero-pad object field (complex/intensity+phase) and save npy.")
    parser.add_argument("--in_complex", type=str, default=None, help="Input complex field .npy (e.g. object_complex.npy)")
    parser.add_argument("--in_intensity", type=str, default=None, help="Input intensity .npy (optional)")
    parser.add_argument("--in_phase", type=str, default=None, help="Input phase .npy (optional)")

    parser.add_argument("--pad", type=int, default=128, help="Padding pixels on each side (default: 128)")
    parser.add_argument("--out_prefix", type=str, default="object_pad_", help="Output prefix (default: object_pad_)")
    parser.add_argument("--show", action="store_true", help="Show visualization")

    args = parser.parse_args()

    field = load_object(args.in_complex, args.in_intensity, args.in_phase)
    field_pad = pad_center(field, args.pad)

    I = np.abs(field) ** 2
    ph = np.angle(field)

    I_pad = np.abs(field_pad) ** 2
    ph_pad = np.angle(field_pad)

    np.save(args.out_prefix + "complex.npy", field_pad)
    np.save(args.out_prefix + "intensity.npy", I_pad)
    np.save(args.out_prefix + "phase.npy", ph_pad)

    print("Saved:")
    print("  ", args.out_prefix + "complex.npy")
    print("  ", args.out_prefix + "intensity.npy")
    print("  ", args.out_prefix + "phase.npy")
    print(f"Original shape: {field.shape} -> Padded shape: {field_pad.shape}")

    if args.show:
        # mask：把低幅值区域相位遮掉，便于观察
        mask = I_pad > (1e-12 * np.max(I_pad))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(I, cmap="gray")
        plt.title("Original intensity")
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(I_pad, cmap="gray")
        plt.title("Padded intensity")
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.show()

        plot_phase(ph, "Original phase [rad]", mask=(I > 1e-12*np.max(I)))
        plot_phase(ph_pad, "Padded phase [rad]", mask=mask)


if __name__ == "__main__":
    main()
