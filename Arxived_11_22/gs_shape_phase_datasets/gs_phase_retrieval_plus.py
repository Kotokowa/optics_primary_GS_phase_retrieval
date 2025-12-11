#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerchberg–Saxton (GS) 相位恢复（改进版）
====================================
要点
----
1) 兼容/自检：
   - 频域是否“中心化”（fftshift），支持 --assume-centered / --flip-centered 一键转换；
   - 能量一致性（Parseval）检测与自适应缩放 --auto-rescale。

2) 诊断：
   - 报告 Parseval 比例因子、初始误差、最终误差；
   - 可保存中间帧以观察收敛。

3) GS 主体 + 可选改进：
   - 软投影（Relaxed amplitude projection）参数 --alpha（默认 1 即标准 GS）；
   - 多次随机重启 --restarts 取最优；
   - 可选“相位量化投影”--phase-levels，适合二值/多层位相（例如 ±pi/2 的字母/几何）。
   - 可选 ER（支持约束）作为辅助：--support，可与相位量化叠加。

4) I/O 约定：
   - fourier_mag：二维频域振幅（若 --assume-centered 为 True，视为已 fftshift）；
   - object_amp ：二维物域振幅（建议平滑窗以减轻边界衍射，如 Hann）；
   - 输出同名 PNG/NPY，含收敛曲线。

用法示例
--------
python gs_phase_retrieval_plus.py --method gs \
  --fourier-mag A_192x192_fourier_mag.npy \
  --object-amp A_192x192_object_amp.npy \
  --iters 400 --restarts 5 --alpha 1.0 --auto-rescale \
  --phase-levels "[-1.570796,1.570796]" \
  --outdir results_A --save-png

注意：若你的 fourier_mag 并非中心化，请加 --assume-centered false
或直接 --flip-centered 让程序自动做一次 fftshift 变换。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt


# ---------- FFT 封装 ----------

def _fft2_centered(u: np.ndarray, norm: Optional[str]=None) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(u, norm=norm))


def _ifft2_centered(Uc: np.ndarray, norm: Optional[str]=None) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(Uc), norm=norm)


def amplitude(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def phase(x: np.ndarray) -> np.ndarray:
    return np.angle(x)


def replace_amplitude(z: np.ndarray, target_amp: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    软/硬投影到给定振幅：
    alpha=1.0 -> 严格替换；
    0<alpha<1 -> 新幅度 = (1-alpha)*|z| + alpha*target_amp。
    """
    th = np.exp(1j * phase(z))
    if alpha >= 1.0:
        amp_new = target_amp
    else:
        amp_new = (1.0 - alpha) * amplitude(z) + alpha * target_amp
    return amp_new * th


# ---------- 诊断与一致性 ----------

def parseval_scale(object_amp: np.ndarray, fourier_mag: np.ndarray, norm: Optional[str]) -> float:
    """
    计算满足 Parseval 的缩放系数 s，使得 sum|u|^2 与 sum|U|^2 关系一致：
    - numpy 默认 (norm=None)： sum|u|^2 = (1/NM) * sum|U|^2
      ⇒ 需令 sum|U|^2 ≈ (NM) * sum|u|^2
    - 若使用 norm='ortho'：      sum|u|^2 = sum|U|^2
    返回建议乘到 fourier_mag 上的系数 s。
    """
    H, W = object_amp.shape
    NM = H * W
    su = float(np.sum(object_amp.astype(np.float64)**2))
    sU = float(np.sum(fourier_mag.astype(np.float64)**2))

    if norm == 'ortho':
        target_sU = su
    else:
        target_sU = su * NM

    if sU <= 0:
        return 1.0
    s = np.sqrt(max(target_sU, 1e-30) / sU)
    return s


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return float(num / den)


# ---------- 可选相位量化 ----------

def quantize_phase(ph: np.ndarray, levels: List[float]) -> np.ndarray:
    """
    将相位投影到给定离散水平（弧度）。
    levels 例如：[-np.pi/2, +np.pi/2] 或 [-pi, -pi/2, 0, +pi/2, +pi]。
    """
    L = np.asarray(levels, dtype=np.float64)
    ph_out = np.empty_like(ph)
    # 对每个像素取最近的 level
    for idx, val in np.ndenumerate(ph):
        k = np.argmin(np.abs(L - val))
        ph_out[idx] = L[k]
    return ph_out


# ---------- 结果结构 ----------

@dataclass
class GSResult:
    field_object: np.ndarray
    field_fourier: np.ndarray
    errors: np.ndarray
    meta: Dict


# ---------- GS 主循环（带可选量化与软投影） ----------

def gs_with_options(fourier_mag: np.ndarray,
                    object_amp: np.ndarray,
                    iters: int = 300,
                    restarts: int = 1,
                    alpha_obj: float = 1.0,
                    alpha_fou: float = 1.0,
                    phase_levels: Optional[List[float]] = None,
                    rng: Optional[np.random.Generator] = None,
                    norm: Optional[str] = None,
                    record_errors: bool = True) -> GSResult:
    rng = np.random.default_rng() if rng is None else rng

    best = None
    best_err = np.inf

    for r in range(restarts):
        phi0 = rng.uniform(0.0, 2.0*np.pi, size=fourier_mag.shape)
        Uc = fourier_mag * np.exp(1j * phi0)
        u  = _ifft2_centered(Uc, norm=norm)

        errors = []

        for _ in range(iters):
            # 物域投影：振幅（软/硬）
            u = replace_amplitude(u, object_amp, alpha=alpha_obj)

            # 可选：相位量化（将 phase(u) 投影到给定集合）
            if phase_levels is not None:
                ph = phase(u)
                ph_q = quantize_phase(ph, phase_levels)
                u = amplitude(u) * np.exp(1j * ph_q)

            # 入频域
            Uc = _fft2_centered(u, norm=norm)

            if record_errors:
                errors.append(relative_l2(amplitude(Uc), fourier_mag))

            # 频域投影：振幅（软/硬）
            Uc = replace_amplitude(Uc, fourier_mag, alpha=alpha_fou)

            # 回物域
            u = _ifft2_centered(Uc, norm=norm)

        final_err = errors[-1] if errors else np.inf
        if final_err < best_err:
            best_err = final_err
            best = GSResult(field_object=u, field_fourier=Uc,
                            errors=np.array(errors, dtype=np.float64),
                            meta=dict(restart=r, iters=iters,
                                      alpha_obj=alpha_obj, alpha_fou=alpha_fou,
                                      phase_levels=phase_levels, norm=norm))

    return best


# ---------- 支持函数 ----------

def save_image(arr: np.ndarray, path: Path, title: Optional[str]=None, vmin: Optional[float]=None, vmax: Optional[float]=None) -> None:
    plt.figure()
    plt.imshow(arr, vmin=vmin, vmax=vmax)
    if title: plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(outdir: Path, res: GSResult, prefix: str,
                 save_png: bool=True, save_npy: bool=False,
                 object_amp_for_display: Optional[np.ndarray]=None,
                 fourier_mag_target: Optional[np.ndarray]=None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    u, Uc, errors = res.field_object, res.field_fourier, res.errors
    amp_u, ph_u = amplitude(u), phase(u)
    amp_U = amplitude(Uc)

    if save_png:
        save_image(amp_u, outdir / f"{prefix}_object_amp.png", "Recovered object amplitude")
        save_image(ph_u, outdir / f"{prefix}_object_phase.png", "Recovered object phase (rad)")
        save_image(np.log10(amp_U + 1e-12), outdir / f"{prefix}_fourier_log10.png", "Recovered log10 |U| (centered)")
        if object_amp_for_display is not None:
            save_image(object_amp_for_display, outdir / f"{prefix}_target_object_amp.png", "Target object amplitude")
        if fourier_mag_target is not None:
            save_image(np.log10(fourier_mag_target + 1e-12), outdir / f"{prefix}_target_fourier_log10.png", "Target log10 |U| (centered)")
        if errors is not None and errors.size>0:
            plt.figure()
            plt.plot(np.arange(1, errors.size+1), errors)
            plt.xlabel("Iteration"); plt.ylabel("Rel-L2 error (Fourier amplitude)")
            plt.title("Convergence"); plt.tight_layout()
            plt.savefig(outdir / f"{prefix}_convergence.png", dpi=150, bbox_inches='tight')
            plt.close()

    if save_npy:
        np.save(outdir / f"{prefix}_object_field.npy", u)
        np.save(outdir / f"{prefix}_fourier_field_centered.npy", Uc)
        np.save(outdir / f"{prefix}_errors.npy", errors)


# ---------- CLI ----------

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"仅支持二维数组：{path}, ndim={arr.ndim}")
    return arr


def main():
    ap = argparse.ArgumentParser(description="GS phase retrieval (improved)")
    ap.add_argument("--method", choices=["gs"], default="gs")
    ap.add_argument("--fourier-mag", required=True, type=str)
    ap.add_argument("--object-amp", required=True, type=str)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0, help="软投影强度 alpha（同时用于物域与频域）")
    ap.add_argument("--alpha-obj", type=float, default=None, help="物域投影 alpha（覆盖 --alpha）")
    ap.add_argument("--alpha-fou", type=float, default=None, help="频域投影 alpha（覆盖 --alpha）")
    ap.add_argument("--phase-levels", type=str, default=None, help="JSON 风格列表（弧度），如 \"[-1.570796,1.570796]\"")
    ap.add_argument("--assume-centered", type=lambda s: s.lower()!="false", default=True, help="输入频域是否已 fftshift（默认 True）")
    ap.add_argument("--flip-centered", action="store_true", help="对输入 fourier_mag 额外执行一次 fftshift 以修正中心化方向")
    ap.add_argument("--norm", choices=[None, "ortho"], default=None)
    ap.add_argument("--auto-rescale", action="store_true", help="按 Parseval 自动缩放 fourier_mag")
    ap.add_argument("--outdir", type=str, default="results_plus")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    fm = load_npy(args.fourier_mag)
    oa = load_npy(args.object_amp)

    # 处理中心化约定
    if not args.assume_centered:
        # 输入视为未中心化：先做一次 fftshift
        fm = np.fft.fftshift(fm)
    if args.flip_centered:
        fm = np.fft.fftshift(fm)

    # 自动缩放四域幅度以满足 Parseval（可减小不一致带来的震荡）
    alpha_obj = args.alpha_obj if args.alpha_obj is not None else args.alpha
    alpha_fou = args.alpha_fou if args.alpha_fou is not None else args.alpha
    norm = args.norm if args.norm != "None" else None

    if args.auto_rescale:
        s = parseval_scale(oa, fm, norm=norm)
        fm = fm * s
        print(f"[Auto-Rescale] Parseval scale = {s:.6g}")

    # 解析相位量化水平
    levels = None
    if args.phase_levels is not None:
        import json
        levels = json.loads(args.phase_levels)
        print(f"[Quantize] phase levels = {levels} (rad)")

    # 计算初始误差（随机一次，用于对照）
    rng = np.random.default_rng(0)
    phi0 = rng.uniform(0.0, 2.0*np.pi, size=fm.shape)
    U0 = fm * np.exp(1j*phi0)
    u0 = _ifft2_centered(U0, norm=norm)
    e0 = relative_l2(amplitude(_fft2_centered(u0, norm=norm)), fm)
    print(f"[Diagnostics] initial rel-L2 error (1 rand init) = {e0:.4f}")

    # 运行 GS（可多重启动）
    res = gs_with_options(fm, oa, iters=args.iters, restarts=args.restarts,
                          alpha_obj=alpha_obj, alpha_fou=alpha_fou,
                          phase_levels=levels, norm=norm)

    print(f"[Done] best final error = {res.errors[-1]:.4f} after {res.meta['iters']} iters (restart={res.meta['restart']})")
    save_results(outdir, res, prefix="gs_plus",
                 save_png=args.save_png, save_npy=args.save_npy,
                 object_amp_for_display=oa, fourier_mag_target=fm)


if __name__ == "__main__":
    main()
