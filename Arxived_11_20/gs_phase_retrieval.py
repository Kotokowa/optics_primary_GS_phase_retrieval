#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerchberg–Saxton (GS) phase retrieval with both real-space and Fourier-space amplitude constraints.

Inputs:
  - real_amp:  path to a 2D array file containing the *real-space amplitude* A(x). Supported: .npy, .npz (key 'arr' or 'data' or 'A'),
               .txt/.csv (whitespace or comma separated).
  - freq_amp:  path to a 2D array file containing the *centered Fourier magnitude* B(k) = |FFT_c{U}|,
               where FFT_c is the centered FFT (fftshift(fft2(ifftshift(.)))) . Supported formats same as above.
  - (optional) val_file: path to a .npz containing a key 'obj_complex' with the target complex object U_target(x) for validation.

Outputs (in --outdir):
  - convergence.csv: iteration, err_k (Fourier-magnitude error), err_x (real-space amplitude error)
  - convergence.png: plot of err_k vs iteration
  - recovered_object_complex.npz: key 'obj_complex' holds complex array U(x)
  - recovered_object_amplitude.png
  - recovered_object_phase.png
  - recovered_freq_complex_centered.npz: key 'freq_complex' holds centered complex spectrum
  - recovered_freq_log_amplitude.png
  - target_freq_log_amplitude.png
  - target_object_amplitude.png
  - A JSON summary printed to stdout at the end (also saved as summary.json)

Notes:
  * All Fourier-domain quantities are CENTERED.
  * The algorithm enforces Fourier magnitude first at each iteration, then real-space amplitude.
  * Stopping criteria: max_iters or |err_k(it) - err_k(it-10)| < tol for 10-iter window (if it>=10).
"""

import argparse, os, sys, json
import numpy as np
import matplotlib.pyplot as plt

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        d = np.load(path)
        for k in ("arr", "data", "A", "B", "obj_complex"):
            if k in d:
                return d[k]
        # fallback: first array
        for k in d.files:
            return d[k]
        raise ValueError("Empty .npz file: %s" % path)
    if ext in (".txt", ".csv"):
        try:
            return np.loadtxt(path, delimiter=None)
        except Exception:
            return np.loadtxt(path, delimiter=",")
    raise ValueError("Unsupported file type for: %s" % path)

def save_complex_npz(path, key, arr):
    np.savez_compressed(path, **{key: arr})

def normalize_eps(x, eps=1e-12):
    return np.maximum(x, eps)

def gs_phase_retrieval(Ax, Bk, max_iters=1000, tol=1e-6, seed=0):
    """
    Ax: real-space amplitude constraint (>=0), shape (N,N)
    Bk: centered Fourier magnitude constraint (>=0), shape (N,N)
    Returns: U (complex object), errors list [(err_k, err_x), ...]
    """
    rng = np.random.default_rng(seed)
    N, M = Ax.shape
    # random initial phase in real-space
    U = Ax * np.exp(1j * rng.uniform(0, 2*np.pi, size=(N, M)))
    errors = []
    Bk_den = np.linalg.norm(Bk)
    Ax_den = np.linalg.norm(Ax)

    for it in range(1, max_iters+1):
        # enforce Fourier magnitude
        F = fft2c(U)
        F_phase = np.exp(1j * np.angle(F))
        F = Bk * F_phase
        # go back to real-space
        U = ifft2c(F)
        # enforce real-space amplitude
        U = Ax * np.exp(1j * np.angle(U))

        # Compute errors
        if it == 1 or it % 1 == 0:
            F_now = fft2c(U)
            err_k = np.linalg.norm(np.abs(F_now) - Bk) / (Bk_den + 1e-15)
            err_x = np.linalg.norm(np.abs(U) - Ax) / (Ax_den + 1e-15)
            errors.append((float(err_k), float(err_x)))

        # simple convergence check using last 10 iterations of err_k
        if len(errors) > 10:
            w = 10
            delta = abs(errors[-1][0] - errors[-w][0])
            if delta < tol:
                break

    return U, errors

def compare_with_target(U, U_tgt, support=None):
    """
    Return amplitude RMSE (on support if provided) and phase MAE after removing global phase.
    """
    if support is None:
        support = np.ones_like(U, dtype=bool)
    # mask
    mask = support & (np.abs(U_tgt) > 1e-12)

    # global phase alignment using inner product
    num = np.vdot(U_tgt[mask], U[mask])
    phi = -np.angle(num)  # phase to multiply U by
    U_aligned = U * np.exp(1j * phi)

    amp_rmse = np.sqrt(np.mean((np.abs(U_aligned[mask]) - np.abs(U_tgt[mask]))**2))
    dphi = np.angle(U_aligned[mask] * np.conj(U_tgt[mask]))
    # wrap to [-pi,pi]; MAE of phase
    phase_mae = np.mean(np.abs(dphi))

    return float(amp_rmse), float(phase_mae), float(phi)

def main():
    parser = argparse.ArgumentParser(description="Gerchberg–Saxton phase retrieval (centered FFT).")
    parser.add_argument("--real_amp", required=True, help="Path to real-space amplitude file (A(x)).")
    parser.add_argument("--freq_amp", required=True, help="Path to centered Fourier magnitude file (B(k)).")
    parser.add_argument("--val_file", default=None, help="Optional .npz with key 'obj_complex' for validation.")
    parser.add_argument("--outdir", default="outputs", help="Directory to save outputs.")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    Ax = load_array(args.real_amp).astype(float)
    Bk = load_array(args.freq_amp).astype(float)

    if Ax.shape != Bk.shape:
        print("Error: shape mismatch: A(x) %s vs B(k) %s" % (Ax.shape, Bk.shape), file=sys.stderr)
        sys.exit(2)

    U, errs = gs_phase_retrieval(Ax, Bk, max_iters=args.max_iters, tol=args.tol, seed=args.seed)

    # Save convergence
    conv_path = os.path.join(args.outdir, "convergence.csv")
    with open(conv_path, "w") as f:
        f.write("iter,err_k,err_x\n")
        for i, (ek, ex) in enumerate(errs, start=1):
            f.write(f"{i},{ek:.8e},{ex:.8e}\n")

    # Plot convergence (err_k)
    its = np.arange(1, len(errs)+1)
    errk = np.array([e[0] for e in errs])
    plt.figure()
    plt.plot(its, errk)
    plt.xlabel("Iteration")
    plt.ylabel("Fourier magnitude error (err_k)")
    plt.title("Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "convergence.png"), dpi=150)
    plt.close()

    # Save recovered object complex field
    save_complex_npz(os.path.join(args.outdir, "recovered_object_complex.npz"), "obj_complex", U)

    # Save recovered object amplitude and phase images
    plt.figure()
    plt.imshow(np.abs(U), origin="upper")
    plt.title("Recovered object amplitude")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "recovered_object_amplitude.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(np.angle(U), origin="upper", cmap="twilight")
    plt.title("Recovered object phase")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "recovered_object_phase.png"), dpi=150)
    plt.close()

    # Save recovered frequency complex (centered) + log amplitude image
    F = fft2c(U)
    save_complex_npz(os.path.join(args.outdir, "recovered_freq_complex_centered.npz"), "freq_complex", F)

    plt.figure()
    plt.imshow(np.log1p(np.abs(F)), origin="upper")
    plt.title("Recovered Fourier log-amplitude (centered)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "recovered_freq_log_amplitude.png"), dpi=150)
    plt.close()

    # Target images for comparison: from inputs
    plt.figure()
    plt.imshow(np.log1p(Bk), origin="upper")
    plt.title("Target Fourier log-amplitude (centered)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "target_freq_log_amplitude.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(Ax, origin="upper")
    plt.title("Target object amplitude")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "target_object_amplitude.png"), dpi=150)
    plt.close()

    summary = {
        "iterations": int(len(errs)),
        "final_err_k": float(errs[-1][0]) if errs else None,
        "final_err_x": float(errs[-1][1]) if errs else None,
    }

    if args.val_file:
        val = np.load(args.val_file)
        key = "obj_complex"
        if key not in val.files:
            print(f"Validation file missing key '{key}'", file=sys.stderr)
        else:
            U_tgt = val[key]
            if U_tgt.shape != U.shape:
                print("Validation shape mismatch", file=sys.stderr)
            else:
                # define support from A(x) > 0
                support = Ax > 1e-12
                amp_rmse, phase_mae, phi = compare_with_target(U, U_tgt, support=support)
                summary.update({
                    "validation": {
                        "amp_rmse": float(amp_rmse),
                        "phase_mae_rad": float(phase_mae),
                        "global_phase_offset_rad": float(phi),
                    }
                })

    # save summary
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
