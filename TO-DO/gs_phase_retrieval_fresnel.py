#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fresnel / lensless Gerchberg–Saxton phase retrieval.

Forward model:
    object-plane complex field U0(x, y)
        |
        |  Fresnel propagation over distance dz
        v
    sensor-plane complex field Uz(x, y) = P_z{U0}

Measurements:
    - |U0| in object plane (object_amp)
    - |Uz| in sensor plane (sensor_amp)

Goal:
    Recover complex U0 (and Uz) from amplitude information via GS iterations.
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Propagation operator (Fresnel transfer function)
# ----------------------------------------------------------------------
def fresnel_propagate(u0: np.ndarray,
                      wavelength: float,
                      dx: float,
                      dz: float,
                      dy: Optional[float] = None) -> np.ndarray:
    """
    Fresnel propagation using transfer function implementation.

    Parameters
    ----------
    u0 : (Ny, Nx) complex ndarray
        Input field in object plane.
    wavelength : float
        Wavelength in meters.
    dx : float
        Sampling pitch in x [m].
    dz : float
        Propagation distance [m], positive for forward, negative for backward.
    dy : float, optional
        Sampling pitch in y [m]. If None, dy = dx.

    Returns
    -------
    uz : complex ndarray
        Propagated field at distance dz.
    """
    u0 = np.asarray(u0, dtype=np.complex128)
    ny, nx = u0.shape
    if dy is None:
        dy = dx

    k = 2.0 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)  # cycles / m
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    # Fresnel transfer function: H(fx, fy) = exp(-j π λ z (fx^2 + fy^2))
    H = np.exp(-1j * np.pi * wavelength * dz * (FX**2 + FY**2))

    U0 = np.fft.fft2(u0)
    Uz = np.fft.ifft2(U0 * H)
    return Uz.astype(np.complex64)


# ----------------------------------------------------------------------
# Utility: complex NMSE (up to global phase)
# ----------------------------------------------------------------------
def complex_nmse(rec: np.ndarray, target: np.ndarray) -> float:
    """
    Compute NMSE between two complex fields, after removing global phase.

    NMSE = ||rec * exp(-j φ0) - target||^2 / ||target||^2
    where φ0 aligns the global phase.
    """
    rec = np.asarray(rec, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if rec.shape != target.shape:
        raise ValueError(f"Shape mismatch: rec {rec.shape} vs target {target.shape}")

    inner = np.vdot(target.ravel(), rec.ravel())
    phi0 = np.angle(inner)  # global phase offset
    rec_aligned = rec * np.exp(-1j * phi0)

    diff = rec_aligned - target
    num = np.linalg.norm(diff.ravel())**2
    den = np.linalg.norm(target.ravel())**2 + 1e-20
    return float(num / den)


# ----------------------------------------------------------------------
# Utility: save 2D real array as PNG with min/max normalization
# ----------------------------------------------------------------------
def save_image(path: str, array: np.ndarray, cmap: str = "viridis") -> None:
    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot save empty image.")
    vmin = np.min(arr)
    vmax = np.max(arr)
    if vmax > vmin:
        arr_norm = (arr - vmin) / (vmax - vmin)
    else:
        arr_norm = np.zeros_like(arr)
    plt.imsave(path, arr_norm, cmap=cmap)


# ----------------------------------------------------------------------
# Fresnel GS core
# ----------------------------------------------------------------------
def gerchberg_saxton_fresnel(object_amp: np.ndarray,
                             sensor_amp: np.ndarray,
                             wavelength: float,
                             dx: float,
                             dz: float,
                             max_iter: int = 200,
                             tol: float = 1e-6,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gerchberg–Saxton for Fresnel (lensless) geometry.

    Parameters
    ----------
    object_amp : (Ny, Nx) ndarray
        Known amplitude in object plane |U0|.
    sensor_amp : (Ny, Nx) ndarray
        Known amplitude in sensor plane |Uz| (sqrt of measured intensity).
    wavelength : float
        Wavelength [m].
    dx : float
        Pixel pitch [m] (assumed same in x and y).
    dz : float
        Distance from object to sensor [m].
    max_iter : int
        Maximum iteration count.
    tol : float
        Relative sensor-amplitude error tolerance for early stopping.
    verbose : bool
        Print iteration log if True.

    Returns
    -------
    object_field : complex ndarray
        Recovered complex field in object plane.
    sensor_field : complex ndarray
        Corresponding complex field in sensor plane.
    errors : float ndarray
        Relative sensor-amplitude error at each iteration.
    """
    object_amp = np.asarray(object_amp, dtype=float)
    sensor_amp = np.asarray(sensor_amp, dtype=float)
    if object_amp.shape != sensor_amp.shape:
        raise ValueError(f"Shape mismatch: object_amp {object_amp.shape} vs sensor_amp {sensor_amp.shape}")

    ny, nx = object_amp.shape

    rng = np.random.default_rng()
    random_phase = np.exp(1j * 2.0 * np.pi * rng.random((ny, nx)))
    object_field = object_amp * random_phase

    target_norm = np.linalg.norm(sensor_amp.ravel()) + 1e-12
    errors = []

    for it in range(max_iter):
        # Forward propagation to sensor plane
        sensor_field = fresnel_propagate(object_field,
                                         wavelength=wavelength,
                                         dx=dx,
                                         dz=dz)

        current_amp = np.abs(sensor_field)
        err = np.linalg.norm((current_amp - sensor_amp).ravel()) / target_norm
        errors.append(err)

        if verbose:
            print(f"Iter {it+1:4d}/{max_iter:4d}  rel sensor-amp error = {err:.6e}")

        if err < tol:
            if verbose:
                print(f"Converged at iteration {it+1}.")
            break

        # Enforce sensor amplitude constraint
        sensor_phase = np.exp(1j * np.angle(sensor_field))
        sensor_field_constrained = sensor_amp * sensor_phase

        # Backward propagation to object plane
        object_field_back = fresnel_propagate(sensor_field_constrained,
                                              wavelength=wavelength,
                                              dx=dx,
                                              dz=-dz)

        # Enforce object amplitude constraint
        object_phase = np.exp(1j * np.angle(object_field_back))
        object_field = object_amp * object_phase

    # Final forward propagation to get consistent sensor_field
    sensor_field = fresnel_propagate(object_field,
                                     wavelength=wavelength,
                                     dx=dx,
                                     dz=dz)

    return object_field, sensor_field, np.asarray(errors, dtype=float)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gerchberg–Saxton phase retrieval with Fresnel (lensless) propagation."
    )

    parser.add_argument("--object_amp", required=True,
                        help="Path to .npy file with object-plane amplitude |U0| (2D float).")
    parser.add_argument("--sensor_amp", required=True,
                        help="Path to .npy file with sensor-plane amplitude |Uz| = sqrt(I) (2D float).")

    parser.add_argument("--wavelength", type=float, required=True,
                        help="Wavelength in meters, e.g. 532e-9.")
    parser.add_argument("--dx", type=float, required=True,
                        help="Pixel pitch in meters, assumed same in x and y (e.g. 6.5e-6).")
    parser.add_argument("--dz", type=float, required=True,
                        help="Propagation distance from object to sensor in meters.")

    parser.add_argument("--target_object_complex", default=None,
                        help="Optional: .npy file with ground-truth complex object field for validation.")
    parser.add_argument("--target_sensor_complex", default=None,
                        help="Optional: .npy file with ground-truth complex sensor field for validation.")

    parser.add_argument("--max_iter", type=int, default=300,
                        help="Maximum number of GS iterations (default: 300).")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Relative sensor-amplitude error tolerance (default: 1e-6).")

    parser.add_argument("--output_dir", default="gs_fresnel_output",
                        help="Directory to save outputs.")
    parser.add_argument("--no_plots", action="store_true",
                        help="If set, do not save PNG plots (only .npy data).")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load amplitudes
    object_amp = np.load(args.object_amp)
    sensor_amp = np.load(args.sensor_amp)
    if object_amp.shape != sensor_amp.shape:
        raise ValueError(f"Shape mismatch: object_amp {object_amp.shape} vs sensor_amp {sensor_amp.shape}")

    print(f"Loaded object amplitude from {args.object_amp}, shape = {object_amp.shape}")
    print(f"Loaded sensor amplitude from {args.sensor_amp}, shape = {sensor_amp.shape}")

    # Optional ground-truth complex fields
    target_obj = None
    target_sens = None
    if args.target_object_complex is not None:
        target_obj = np.load(args.target_object_complex)
        if target_obj.shape != object_amp.shape:
            raise ValueError(
                f"target_object_complex shape {target_obj.shape} "
                f"does not match amplitude shape {object_amp.shape}"
            )
        print(f"Loaded target complex object field from {args.target_object_complex}")

    if args.target_sensor_complex is not None:
        target_sens = np.load(args.target_sensor_complex)
        if target_sens.shape != sensor_amp.shape:
            raise ValueError(
                f"target_sensor_complex shape {target_sens.shape} "
                f"does not match sensor_amp shape {sensor_amp.shape}"
            )
        print(f"Loaded target complex sensor field from {args.target_sensor_complex}")

    print("Starting Fresnel GS iterations...")
    obj_rec, sens_rec, errors = gerchberg_saxton_fresnel(
        object_amp=object_amp,
        sensor_amp=sensor_amp,
        wavelength=args.wavelength,
        dx=args.dx,
        dz=args.dz,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
    )

    # Save numerical results
    np.save(os.path.join(args.output_dir, "reconstructed_object_complex.npy"),
            obj_rec.astype(np.complex64))
    np.save(os.path.join(args.output_dir, "reconstructed_sensor_complex.npy"),
            sens_rec.astype(np.complex64))
    np.save(os.path.join(args.output_dir, "error_history.npy"),
            errors.astype(float))

    # Plots
    if not args.no_plots:
        # 1) Convergence curve
        plt.figure()
        plt.semilogy(errors)
        plt.xlabel("Iteration")
        plt.ylabel("Relative sensor amplitude error")
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "convergence_curve.png"), dpi=150)
        plt.close()

        # 2) Object amplitude images (target vs reconstructed)
        save_image(os.path.join(args.output_dir, "target_object_amplitude.png"),
                   np.abs(object_amp))
        save_image(os.path.join(args.output_dir, "reconstructed_object_amplitude.png"),
                   np.abs(obj_rec))

        # 3) Sensor amplitude images (target vs reconstructed)
        save_image(os.path.join(args.output_dir, "target_sensor_amplitude.png"),
                   np.abs(sensor_amp))
        save_image(os.path.join(args.output_dir, "reconstructed_sensor_amplitude.png"),
                   np.abs(sens_rec))

        # 4) Object phase images
        save_image(os.path.join(args.output_dir, "reconstructed_object_phase.png"),
                   np.angle(obj_rec))
        if target_obj is not None:
            save_image(os.path.join(args.output_dir, "target_object_phase_from_file.png"),
                       np.angle(target_obj))

        # 5) Sensor phase images (only if target is provided, for comparison)
        if target_sens is not None:
            save_image(os.path.join(args.output_dir, "target_sensor_phase_from_file.png"),
                       np.angle(target_sens))
            save_image(os.path.join(args.output_dir, "reconstructed_sensor_phase.png"),
                       np.angle(sens_rec))

    # Validation NMSEs
    lines = []
    if target_obj is not None:
        nmse_obj = complex_nmse(obj_rec, target_obj)
        print(f"Object-plane complex NMSE = {nmse_obj:.6e}")
        lines.append(f"Object-plane complex NMSE = {nmse_obj:.6e}")
    if target_sens is not None:
        nmse_sens = complex_nmse(sens_rec, target_sens)
        print(f"Sensor-plane complex NMSE = {nmse_sens:.6e}")
        lines.append(f"Sensor-plane complex NMSE = {nmse_sens:.6e}")

    if lines:
        with open(os.path.join(args.output_dir, "validation_error.txt"),
                  "w",
                  encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\\n")

    print("Done.")


if __name__ == "__main__":
    main()
