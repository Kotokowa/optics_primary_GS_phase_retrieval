#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def save_phase_png(path: str, phase: np.ndarray) -> None:
    """把相位（弧度）按 min/max 归一化后，用 viridis 保存成 PNG。"""
    phase = np.asarray(phase, dtype=float)
    vmin = phase.min()
    vmax = phase.max()
    if vmax > vmin:
        x = (phase - vmin) / (vmax - vmin)
    else:
        x = np.zeros_like(phase)
    plt.imsave(path, x, cmap="viridis")


def make_object_field(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """由振幅 + 相位构造物面复场。"""
    amp = np.asarray(amp, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if amp.shape != phase.shape:
        raise ValueError(f"Amplitude shape {amp.shape} and phase shape {phase.shape} must match.")
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def fft_geometry(U_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fourier 平面几何：简单 2D FFT。"""
    F = np.fft.fft2(U_obj)
    return F, np.abs(F)


def fresnel_propagation(
    U0: np.ndarray,
    wavelength: float,
    dx: float,
    dz: float,
) -> np.ndarray:
    """利用角谱法从 z=0 传播到 z=dz。"""
    ny, nx = U0.shape
    k = 2.0 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)  # cycles / m
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    f2 = FX**2 + FY**2

    # 角谱传递函数 H(fx,fy;dz)
    inside = np.maximum(0.0, 1.0 - (wavelength**2) * f2)
    H = np.exp(1j * k * dz * np.sqrt(inside))

    U0_f = np.fft.fft2(U0)
    Uz = np.fft.ifft2(U0_f * H)
    return Uz.astype(np.complex64)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic GS test set from arbitrary object amplitude + phase."
    )
    parser.add_argument("--amp_npy", required=True,
                        help="Path to .npy file with object-plane amplitude (2D float).")
    parser.add_argument("--phase_npy", required=True,
                        help="Path to .npy file with object-plane phase in radians (2D float, same shape).")
    parser.add_argument("--geometry", choices=["fourier", "fresnel"], default="fourier",
                        help="Forward model: 'fourier' (FFT to Fourier plane) or 'fresnel' (free-space propagation).")
    parser.add_argument("--wavelength", type=float, default=532e-9,
                        help="Wavelength in meters (for fresnel). Default: 532 nm.")
    parser.add_argument("--dx", type=float, default=6.5e-6,
                        help="Pixel pitch in meters (for fresnel). Default: 6.5 um.")
    parser.add_argument("--dz", type=float, default=0.1,
                        help="Propagation distance in meters (for fresnel). Default: 0.1 m.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save generated files.")
    parser.add_argument("--prefix", default="obj",
                        help="Prefix for saved file names (default: 'obj').")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amp = np.load(args.amp_npy)
    phase = np.load(args.phase_npy)
    U_obj = make_object_field(amp, phase)

    # 保存物面数据
    np.save(out_dir / f"{args.prefix}_object_amplitude.npy",
            np.abs(amp).astype(np.float32))
    np.save(out_dir / f"{args.prefix}_target_object_complex.npy", U_obj)
    save_phase_png(str(out_dir / f"{args.prefix}_target_phase.png"), phase)

    if args.geometry == "fourier":
        # Fourier 平面（透镜焦平面）
        F, F_amp = fft_geometry(U_obj)
        np.save(out_dir / f"{args.prefix}_fourier_amplitude.npy",
                F_amp.astype(np.float32))
        np.save(out_dir / f"{args.prefix}_target_fourier_complex.npy", F)
    else:
        # Fresnel 自由空间传播到传感器
        Uz = fresnel_propagation(U_obj, args.wavelength, args.dx, args.dz)
        I = np.abs(Uz) ** 2
        A = np.abs(Uz)
        np.save(out_dir / f"{args.prefix}_sensor_amplitude.npy",
                A.astype(np.float32))
        np.save(out_dir / f"{args.prefix}_sensor_intensity.npy",
                I.astype(np.float32))
        np.save(out_dir / f"{args.prefix}_target_sensor_complex.npy", Uz)

    print(f"Saved test set to {out_dir}")
    if args.geometry == "fourier":
        print("Files:")
        print(f"  {args.prefix}_object_amplitude.npy")
        print(f"  {args.prefix}_target_object_complex.npy")
        print(f"  {args.prefix}_fourier_amplitude.npy")
        print(f"  {args.prefix}_target_fourier_complex.npy")
        print(f"  {args.prefix}_target_phase.png")
    else:
        print("Files:")
        print(f"  {args.prefix}_object_amplitude.npy")
        print(f"  {args.prefix}_target_object_complex.npy")
        print(f"  {args.prefix}_sensor_amplitude.npy")
        print(f"  {args.prefix}_sensor_intensity.npy")
        print(f"  {args.prefix}_target_sensor_complex.npy")
        print(f"  {args.prefix}_target_phase.png")


if __name__ == "__main__":
    main()
