#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def gerchberg_saxton(object_amp, fourier_amp, max_iter=200, tol=1e-6, verbose=True):
    """Run the Gerchberg-Saxton phase retrieval algorithm.

    Parameters
    ----------
    object_amp : 2D ndarray
        Known amplitude in object (real) space.
    fourier_amp : 2D ndarray
        Known amplitude in Fourier (frequency) space.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Relative Fourier-amplitude error tolerance for early stopping.
    verbose : bool
        If True, print per-iteration errors.

    Returns
    -------
    object_field : 2D complex ndarray
        Recovered complex field in object space.
    fourier_field : 2D complex ndarray
        Corresponding complex field in Fourier space.
    errors : 1D ndarray of float
        Relative Fourier-amplitude error at each iteration.
    """
    object_amp = np.asarray(object_amp, dtype=float)
    fourier_amp = np.asarray(fourier_amp, dtype=float)
    if object_amp.shape != fourier_amp.shape:
        raise ValueError(f"Shape mismatch: object_amp {object_amp.shape} vs fourier_amp {fourier_amp.shape}")

    ny, nx = object_amp.shape
    rng = np.random.default_rng()
    random_phase = np.exp(1j * 2.0 * np.pi * rng.random((ny, nx)))
    object_field = object_amp * random_phase

    target_norm = np.linalg.norm(fourier_amp.ravel()) + 1e-12
    errors = []

    for it in range(max_iter):
        fourier_field = np.fft.fft2(object_field)
        current_amp = np.abs(fourier_field)
        err = np.linalg.norm((current_amp - fourier_amp).ravel()) / target_norm
        errors.append(err)
        if verbose:
            print(f"Iteration {it + 1:4d}/{max_iter:4d}  relative Fourier amplitude error = {err:.6e}")
        if err < tol:
            if verbose:
                print(f"Converged at iteration {it + 1}")
            break

        # Enforce Fourier amplitude constraint
        fourier_phase = np.exp(1j * np.angle(fourier_field))
        fourier_field_constrained = fourier_amp * fourier_phase

        # Back to object plane
        object_field_back = np.fft.ifft2(fourier_field_constrained)
        object_phase = np.exp(1j * np.angle(object_field_back))
        object_field = object_amp * object_phase

    # Final Fourier field corresponding to last object_field
    fourier_field = np.fft.fft2(object_field)
    return object_field, fourier_field, np.asarray(errors, dtype=float)


def align_reconstruction_to_target(rec, target):
    """把重建复场 rec 对齐到 target：

    步骤：
    1) 先对齐一个整体相位因子；
    2) 用 FFT 计算复数互相关，估计最佳整像素平移 (dy, dx)；
    3) 做循环平移，再细调一次整体相位；
    4) 返回对齐后的复场 + 平移量 + 总相位因子 + 对齐后的 NMSE。

    注意：这里的 NMSE 是在不再额外去掉整体相位的情况下算的，
          只是供打印参考。主程序最后仍用 compute_complex_nmse 做标准评估。
    """
    rec = np.asarray(rec, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if rec.shape != target.shape:
        raise ValueError(f"Shape mismatch in alignment: rec {rec.shape} vs target {target.shape}")

    ny, nx = rec.shape

    # 1) 先对齐整体相位：rec1 = rec * e^{-j phi1}
    inner1 = np.vdot(target.ravel(), rec.ravel())
    phi1 = np.angle(inner1)
    rec1 = rec * np.exp(-1j * phi1)

    # 2) 用互相关估计循环平移 (dy, dx)
    F_t = np.fft.fft2(target)
    F_r = np.fft.fft2(rec1)
    C = np.fft.ifft2(F_t * np.conj(F_r))  # 复数互相关（循环）
    peak_idx = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    y0, x0 = peak_idx

    # 把 [0, ny-1] 的索引换成带符号的位移
    dy = y0 if y0 <= ny // 2 else y0 - ny
    dx = x0 if x0 <= nx // 2 else x0 - nx

    # 在空间域做循环平移（正 dy = 向下移，正 dx = 向右移）
    rec2 = np.roll(np.roll(rec1, dy, axis=0), dx, axis=1)

    # 3) 再细调一次整体相位：rec3 = rec2 * e^{-j phi2}
    inner2 = np.vdot(target.ravel(), rec2.ravel())
    phi2 = np.angle(inner2)
    rec3 = rec2 * np.exp(-1j * phi2)

    total_phase = phi1 + phi2

    # 4) 对齐后的 NMSE（不再额外去相位）
    diff = rec3 - target
    num = np.linalg.norm(diff.ravel()) ** 2
    den = np.linalg.norm(target.ravel()) ** 2 + 1e-20
    nmse_aligned = float(num / den)

    return rec3, (dy, dx), total_phase, nmse_aligned



def compute_complex_nmse(reconstructed, target):
    """Compute normalized MSE between two complex fields, up to a global phase factor."""
    reconstructed = np.asarray(reconstructed, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if reconstructed.shape != target.shape:
        raise ValueError(f"Shape mismatch between reconstructed {reconstructed.shape} and target {target.shape}")

    inner = np.vdot(target.ravel(), reconstructed.ravel())
    global_phase = np.angle(inner)
    reconstructed_aligned = reconstructed * np.exp(-1j * global_phase)
    diff = reconstructed_aligned - target
    num = np.linalg.norm(diff.ravel()) ** 2
    den = np.linalg.norm(target.ravel()) ** 2 + 1e-20
    return float(num / den)


def save_image(path, array, normalize=True, vmin=None, vmax=None, cmap='viridis'):
    """保存 2D 实数数组为 PNG。

    参数
    ----
    path : str
        输出文件路径。
    array : 2D array-like
        要保存的数据。
    normalize : bool
        True: 先按 [vmin, vmax] 线性归一化到 [0,1] 再保存（类似原来的行为）。
        False: 不做归一化，直接用给定的 vmin/vmax 作为色标范围保存。
    vmin, vmax : float 或 None
        映射到 colormap 的最小/最大值。为 None 时自动用数据本身的 min/max。
    cmap : str
        Matplotlib colormap 名称。
    """
    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot save empty image")

    if normalize:
        # 和原来类似的归一化，但允许手动指定 vmin/vmax
        if vmin is None:
            vmin = np.min(arr)
        if vmax is None:
            vmax = np.max(arr)
        if vmax > vmin:
            arr_norm = (arr - vmin) / (vmax - vmin)
        else:
            arr_norm = np.zeros_like(arr)
        plt.imsave(path, arr_norm, cmap=cmap)
    else:
        # 不做归一化，直接使用 vmin/vmax 映射到 colormap
        if vmin is None:
            vmin = np.min(arr)
        if vmax is None:
            vmax = np.max(arr)
        plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def save_image_with_colorbar(path, array, vmin=None, vmax=None, cmap='viridis'):
    """保存带 colorbar 的图像（不归一化，使用 vmin/vmax 做色标）。

    适合用来画振幅 / 相位，直观看出数值范围。
    """
    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot save empty image")

    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)

    plt.figure()
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="Gerchberg-Saxton phase retrieval")
    parser.add_argument('--object_amp', required=True,
                        help='Path to .npy file containing object-space amplitude (float, 2D).')
    parser.add_argument('--fourier_amp', required=True,
                        help='Path to .npy file containing Fourier-space amplitude (float, 2D).')
    parser.add_argument('--target_complex', default=None,
                        help='Optional: path to .npy file with target complex object field.')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum number of iterations (default: 300).')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Relative Fourier-amplitude tolerance (default: 1e-6).')
    parser.add_argument('--output_dir', default='gs_output',
                        help='Directory to store results (will be created).')
    parser.add_argument('--no_plots', action='store_true',
                        help='If set, only save .npy data (no PNGs).')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    object_amp = np.load(args.object_amp)
    fourier_amp = np.load(args.fourier_amp)
    print(f'Loaded object_amp from {args.object_amp}, shape = {object_amp.shape}')
    print(f'Loaded fourier_amp from {args.fourier_amp}, shape = {fourier_amp.shape}')
    if object_amp.shape != fourier_amp.shape:
        raise ValueError(f'Shape mismatch between object_amp {object_amp.shape} and fourier_amp {fourier_amp.shape}')

    if args.target_complex is not None:
        target_complex = np.load(args.target_complex)
        if target_complex.shape != object_amp.shape:
            raise ValueError(f'Target complex field shape {target_complex.shape} does not match amplitude shape {object_amp.shape}')
        print(f'Loaded target complex object field from {args.target_complex}')
    else:
        target_complex = None

    print('Starting Gerchberg-Saxton iterations...')
    obj_rec, F_rec, errors = gerchberg_saxton(object_amp, fourier_amp,
                                               max_iter=args.max_iter,
                                               tol=args.tol,
                                               verbose=True)

    # 先保留一份“原始”的重建结果（仅 GS 迭代，不做任何对齐）
    obj_rec_raw = obj_rec

    # 如果给了 target_complex，就把重建结果对齐到 target（整体相位 + 整像素平移）
    if target_complex is not None:
        print('Aligning reconstructed object field to target (global phase + integer shift)...')
        obj_rec_aligned, shift, total_phase, nmse_aligned = align_reconstruction_to_target(obj_rec_raw, target_complex)
        print(f'  Estimated shift (dy, dx) = {shift}')
        print(f'  NMSE after alignment (no extra phase removal) = {nmse_aligned:.6e}')
        obj_rec = obj_rec_aligned
    else:
        print('No target_complex provided: using raw GS reconstruction without alignment.')

    # Save core numerical results
    # 原始 GS 结果
    np.save(os.path.join(args.output_dir, 'reconstructed_object_complex_raw.npy'),
            obj_rec_raw.astype(np.complex64))
    # 对齐后的结果（后续画图和评估都用这个）
    np.save(os.path.join(args.output_dir, 'reconstructed_object_complex.npy'),
            obj_rec.astype(np.complex64))

    np.save(os.path.join(args.output_dir, 'reconstructed_fourier_complex.npy'), F_rec.astype(np.complex64))
    F_rec_centered = np.fft.fftshift(F_rec)
    np.save(os.path.join(args.output_dir, 'reconstructed_fourier_complex_centered.npy'), F_rec_centered.astype(np.complex64))
    np.save(os.path.join(args.output_dir, 'error_history.npy'), np.asarray(errors, dtype=float))
    

    if not args.no_plots:
        # Convergence curve
        plt.figure()
        plt.semilogy(errors)
        plt.xlabel('Iteration')
        plt.ylabel('Relative Fourier amplitude error')
        plt.grid(True, which='both', linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'convergence_curve.png'), dpi=150)
        plt.close()

        # Target Fourier log-amplitude (centered)
        Famp_target_centered = np.fft.fftshift(np.abs(fourier_amp))
        log_Famp_target = np.log1p(Famp_target_centered)
        save_image(os.path.join(args.output_dir, 'target_fourier_log_amplitude.png'),
                   log_Famp_target, normalize=True)

        # Reconstructed Fourier log-amplitude (centered)
        Famp_rec_centered = np.fft.fftshift(np.abs(F_rec))
        log_Famp_rec = np.log1p(Famp_rec_centered)
        save_image(os.path.join(args.output_dir, 'reconstructed_fourier_log_amplitude.png'),
                   log_Famp_rec, normalize=True)

        # ---------- 实空间振幅：用统一的 [0, 1] 色标 + colorbar ----------
        amp_target = np.abs(object_amp)
        amp_rec = np.abs(obj_rec)
        # 对于你的数据，这两个几乎都是 1，这里固定 0~1 更直观
        amp_vmin, amp_vmax = 0.0, 3.0

        save_image_with_colorbar(
            os.path.join(args.output_dir, 'target_object_amplitude.png'),
            amp_target, vmin=amp_vmin, vmax=amp_vmax, cmap='viridis'
        )
        save_image_with_colorbar(
            os.path.join(args.output_dir, 'reconstructed_object_amplitude.png'),
            amp_rec, vmin=amp_vmin, vmax=amp_vmax, cmap='viridis'
        )

        # ---------- 实空间相位：对齐后画出来 ----------
        phase_rec = np.angle(obj_rec)

        if target_complex is not None:
            target_phase = np.angle(target_complex)
            # 用一个共同的色标范围，保证两张图颜色可比
            phase_min = float(min(phase_rec.min(), target_phase.min()))
            phase_max = float(max(phase_rec.max(), target_phase.max()))

            save_image_with_colorbar(
                os.path.join(args.output_dir, 'target_object_phase_from_file.png'),
                target_phase, vmin=phase_min, vmax=phase_max, cmap='twilight'
            )
            save_image_with_colorbar(
                os.path.join(args.output_dir, 'reconstructed_object_phase.png'),
                phase_rec, vmin=phase_min, vmax=phase_max, cmap='twilight'
            )
        else:
            # 没有 target 时，至少把重建相位画出来
            phase_min = float(phase_rec.min())
            phase_max = float(phase_rec.max())
            save_image_with_colorbar(
                os.path.join(args.output_dir, 'reconstructed_object_phase.png'),
                phase_rec, vmin=phase_min, vmax=phase_max, cmap='twilight'
            )

    # Validation against target complex field
    if target_complex is not None:
        nmse = compute_complex_nmse(obj_rec, target_complex)
        print(f'Validation against target complex field: complex NMSE = {nmse:.6e}')
        with open(os.path.join(args.output_dir, 'validation_error.txt'), 'w', encoding='utf-8') as f:
            f.write('Complex NMSE (after removing global phase) = {:.6e}\n'.format(nmse))

    print('Done.')


if __name__ == '__main__':
    main()
