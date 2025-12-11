#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerchberg–Saxton (GS) 相位恢复程序（含 Fienup Error-Reduction 变体）
===================================================================
作者：ChatGPT（GPT-5 Pro）
许可：MIT

功能概览
--------
1. 经典 GS：已知“物面（object plane）振幅”和“频域（Fourier plane）振幅”，
   交替投影恢复物面相位（或得到能满足两域振幅约束的复场）。
2. ER 变体（Fienup Error-Reduction）：仅已知频域振幅 + 物域支持（support）/非负等约束。

命令行示例
----------
# 运行 GS：读取 .npy 文件（均为 float32/64 的二维 numpy 数组）
python gs_phase_retrieval.py --method gs \
    --fourier_mag data/fourier_mag.npy \
    --object_amp data/object_amp.npy \
    --iters 300 --seed 0 --outdir results_gs --save-png --save-npy

# 运行 ER：仅有频域振幅 + 支持约束（可选非负/实数约束）
python gs_phase_retrieval.py --method er \
    --fourier_mag data/fourier_mag.npy \
    --support data/support_mask.npy \
    --positivity --enforce-real \
    --iters 1000 --seed 0 --outdir results_er --save-png --save-npy

# 运行内置演示（无需外部数据），生成合成目标并恢复
python gs_phase_retrieval.py --demo gs --iters 250 --outdir demo_gs --save-png
python gs_phase_retrieval.py --demo er --iters 800 --outdir demo_er --save-png --positivity --enforce-real

输入 / 输出约定
--------------
- fourier_mag：二维数组，表示中心化后的频域振幅（DC 在中心）。
- object_amp：二维数组，表示物域振幅（与 fourier_mag 形状一致）。
- support   ：二维 0/1 掩膜，1 表示物体所在区域（支持区域）。
- 程序会在 --outdir 下保存若干 PNG/Numpy 文件（可通过 --save-png / --save-npy 控制）。

原理梳理
--------
GS（两域振幅已知）核心迭代：
(1) 从当前物域复场 u(x) 做 FFT 得到频域 U(f)，把其振幅替换为已知的 |Û(f)|，保留相位 → U'(f)。
(2) iFFT 回物域得到 u'(x)，再把其振幅替换为已知的 |û(x)|，保留相位 → u(x)。
往复迭代，直至误差收敛。误差常以频域振幅相对偏差度量。

ER（仅频域振幅 + 物域先验）核心迭代：
(1) 频域投影同上：替换为已知 |Û(f)|，保留相位；
(2) 回到物域后，应用先验：支持外设零（support）、实值约束、非负约束等。
该方法是 Fienup 家族中最基础的“错误约减（Error-Reduction）”算法。

复杂度
------
每次迭代主要代价是 2 次 FFT（一次正变换 + 一次逆变换），复杂度约 O(N log N)。

注意事项
--------
- 输入尺寸最好为 2 的幂或合适的合数以加速 FFT。
- 适度零填充/窗口可改善边界影响；多随机初值可减少陷入坏局部极小值概率。
- 物域/频域的“中心化”约定必须一致。此处统一在“中心化的频域（fftshift）”上施加振幅约束。

"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict


# ============ 工具函数（FFT 包装，误差计算，I/O） ============

def _fft2_centered(u: np.ndarray) -> np.ndarray:
    """中心化 FFT：返回 fftshift(fft2(u))，DC 在中心。"""
    return np.fft.fftshift(np.fft.fft2(u))


def _ifft2_centered(Uc: np.ndarray) -> np.ndarray:
    """中心化 iFFT：输入为中心化频谱（DC 在中心），先 ifftshift 再 ifft2。"""
    return np.fft.ifft2(np.fft.ifftshift(Uc))


def amplitude(x: np.ndarray) -> np.ndarray:
    """复数数组的振幅 |x|。"""
    return np.abs(x)


def phase(x: np.ndarray) -> np.ndarray:
    """复数数组的相位 ∠x（弧度，范围 -π~π）。"""
    return np.angle(x)


def replace_amplitude(z: np.ndarray, target_amp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    将复数数组 z 的振幅替换为 target_amp，保持相位不变。
    避免 0 除：对 z 的振幅加 eps。
    """
    ph = np.exp(1j * phase(z))
    return target_amp * ph


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """相对 L2 误差：||a-b||_2 / (||b||_2 + eps)。"""
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return float(num / den)


def ensure_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """保证两个数组形状一致。"""
    if a.shape != b.shape:
        raise ValueError(f"{name_a} 形状 {a.shape} 与 {name_b} 形状 {b.shape} 不一致。")


def save_image(arr: np.ndarray, path: Path, title: Optional[str] = None, vmin: Optional[float]=None, vmax: Optional[float]=None) -> None:
    """
    保存单图像（不指定颜色映射，遵循默认设置）。
    注意：依照要求，每个图单独一个 figure，不使用子图且不指定颜色。
    """
    plt.figure()
    plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.title(title if title is not None else path.stem)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ============ 数据类：结果包 ============

@dataclass
class GSResult:
    field_object: np.ndarray     # 物域复场（复杂数）
    field_fourier: np.ndarray    # 频域复场（中心化）
    errors: np.ndarray           # 误差曲线（频域振幅相对误差）
    meta: Dict                   # 元信息（迭代数、seed 等）


# ============ 核心算法 ============

def gerchberg_saxton(fourier_mag: np.ndarray,
                     object_amp: np.ndarray,
                     iters: int = 200,
                     init_phase: Optional[np.ndarray] = None,
                     seed: Optional[int] = None,
                     record_errors: bool = True) -> GSResult:
    """
    经典 GS：两域振幅已知。
    参数
    ----
    fourier_mag : 中心化的频域振幅 |Û(f)|（二维）
    object_amp  : 物域振幅 |û(x)|（二维）
    iters       : 迭代次数
    init_phase  : 初始频域相位（二维，弧度），若 None 则随机
    seed        : 随机种子（用于随机初相位）
    record_errors : 是否记录每次迭代的频域振幅相对误差

    返回
    ----
    GSResult：含最终物域/频域复场、误差曲线等。
    """
    fourier_mag = np.asarray(fourier_mag, dtype=np.float64)
    object_amp = np.asarray(object_amp, dtype=np.float64)
    ensure_same_shape(fourier_mag, object_amp, "fourier_mag", "object_amp")

    rng = np.random.default_rng(seed)

    # 初始化频域复场：幅度为测量 fourier_mag，相位为给定/随机
    if init_phase is None:
        phi0 = rng.uniform(0.0, 2.0 * np.pi, size=fourier_mag.shape)
    else:
        phi0 = np.asarray(init_phase, dtype=np.float64)
        ensure_same_shape(phi0, fourier_mag, "init_phase", "fourier_mag")

    Uc = fourier_mag * np.exp(1j * phi0)  # 中心化频域复场
    u = _ifft2_centered(Uc)               # 回到物域

    errors = []

    for _ in range(iters):
        # 物域投影：替换振幅为 object_amp，保留相位
        u = replace_amplitude(u, object_amp)

        # 进频域
        Uc = _fft2_centered(u)

        # 记录误差（与目标频域振幅的相对 L2 偏差）
        if record_errors:
            err = relative_l2(amplitude(Uc), fourier_mag)
            errors.append(err)

        # 频域投影：替换振幅为 fourier_mag，保留相位
        Uc = replace_amplitude(Uc, fourier_mag)

        # 回物域
        u = _ifft2_centered(Uc)

    return GSResult(
        field_object=u,
        field_fourier=Uc,
        errors=np.array(errors, dtype=np.float64),
        meta=dict(method="GS", iters=iters, seed=seed)
    )


def fienup_error_reduction(fourier_mag: np.ndarray,
                           support: Optional[np.ndarray] = None,
                           iters: int = 500,
                           seed: Optional[int] = None,
                           positivity: bool = False,
                           enforce_real: bool = False,
                           record_errors: bool = True) -> GSResult:
    """
    Fienup Error-Reduction（ER）算法：仅已知频域振幅 + 物域先验。
    参数
    ----
    fourier_mag : 中心化频域振幅 |Û(f)|（二维）
    support     : 物域支持掩膜（0/1），形状同 fourier_mag；None 表示无支持约束
    iters       : 迭代次数
    seed        : 随机种子（用于随机初相位）
    positivity  : 是否在物域施加非负约束（仅在 enforce_real=True 时有效）
    enforce_real: 是否施加实值约束（将物域复场取实部）
    record_errors : 是否记录误差曲线

    返回
    ----
    GSResult：含最终物域/频域复场、误差曲线等。
    """
    fourier_mag = np.asarray(fourier_mag, dtype=np.float64)
    if support is not None:
        support = np.asarray(support, dtype=np.float64)
        ensure_same_shape(fourier_mag, support, "fourier_mag", "support")

    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=fourier_mag.shape)
    Uc = fourier_mag * np.exp(1j * phi0)   # 初始频域复场
    u = _ifft2_centered(Uc)                 # 物域

    errors = []

    for _ in range(iters):
        # === 物域先验投影 ===
        if enforce_real:
            u = np.real(u)

        if positivity and enforce_real:
            # 非负约束：负值设为 0
            u = np.maximum(u, 0.0)

        if support is not None:
            # 支持外强制为 0（常见的 ER 物域投影）
            u = u * support

        # === 进频域 ===
        Uc = _fft2_centered(u)

        if record_errors:
            err = relative_l2(amplitude(Uc), fourier_mag)
            errors.append(err)

        # === 频域投影：强制频域振幅 ===
        Uc = replace_amplitude(Uc, fourier_mag)

        # === 回物域 ===
        u = _ifft2_centered(Uc)

    return GSResult(
        field_object=u,
        field_fourier=Uc,
        errors=np.array(errors, dtype=np.float64),
        meta=dict(method="ER", iters=iters, seed=seed, positivity=positivity, enforce_real=enforce_real)
    )


# ============ 演示数据生成 ============

def make_synthetic_object(shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    生成一个简单的物域振幅（二维非负图案），用于演示。
    组合了圆环 + 高斯峰 + 矩形窗，避免依赖外部图像库。
    """
    H, W = shape
    y = np.linspace(-1, 1, H, endpoint=False)
    x = np.linspace(-1, 1, W, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # 圆环
    R = np.sqrt(X**2 + Y**2)
    ring = np.exp(-((R - 0.5) / 0.06) ** 2)

    # 两个高斯点
    g1 = np.exp(-((X + 0.35) ** 2 + (Y + 0.2) ** 2) / (2 * 0.04 ** 2))
    g2 = np.exp(-((X - 0.25) ** 2 + (Y - 0.3) ** 2) / (2 * 0.06 ** 2))

    # 矩形窗
    rect = ((np.abs(X) < 0.6) & (np.abs(Y) < 0.4)).astype(float)

    amp = 0.6 * ring + 0.9 * g1 + 0.7 * g2
    amp *= rect
    amp = amp / (amp.max() + 1e-12)
    return amp


def make_support_from_amp(amp: np.ndarray, thresh: float = 0.08) -> np.ndarray:
    """由振幅阈值生成支持掩膜。"""
    sup = (amp > float(thresh)).astype(float)
    return sup


# ============ 可视化与保存 ============

def save_results(outdir: Path,
                 result: GSResult,
                 save_png: bool = True,
                 save_npy: bool = False,
                 prefix: str = "result",
                 object_amp_for_display: Optional[np.ndarray] = None,
                 fourier_mag_target: Optional[np.ndarray] = None) -> None:
    """
    根据结果包保存图像/数据。
    - object_amp_for_display：若提供会同时保存对比图。
    - fourier_mag_target    ：若提供会保存目标频域振幅（对比误差）。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 物域结果：振幅与相位
    u = result.field_object
    amp_u = amplitude(u)
    ph_u = phase(u)

    # 频域结果：振幅（中心化）
    Uc = result.field_fourier
    amp_U = amplitude(Uc)

    # 误差曲线
    errors = result.errors

    if save_png:
        save_image(amp_u, outdir / f"{prefix}_object_amp.png", title="Recovered object amplitude")
        # 相位范围 -π~π，直接显示
        save_image(ph_u, outdir / f"{prefix}_object_phase.png", title="Recovered object phase")

        # 频域振幅（对数显示以便观察动态范围）
        amp_U_log = np.log10(amp_U + 1e-12)
        save_image(amp_U_log, outdir / f"{prefix}_fourier_mag_log.png", title="Recovered Fourier magnitude (log10)")

        if object_amp_for_display is not None:
            save_image(object_amp_for_display, outdir / f"{prefix}_target_object_amp.png", title="Target object amplitude")

        if fourier_mag_target is not None:
            tgt_log = np.log10(fourier_mag_target + 1e-12)
            save_image(tgt_log, outdir / f"{prefix}_target_fourier_mag_log.png", title="Target Fourier magnitude (log10)")

        # 误差曲线
        if errors.size > 0:
            plt.figure()
            plt.plot(np.arange(1, errors.size + 1), errors)
            plt.xlabel("Iteration")
            plt.ylabel("Relative L2 error (Fourier amplitude)")
            plt.title("Convergence")
            plt.tight_layout()
            plt.savefig(outdir / f"{prefix}_convergence.png", dpi=150, bbox_inches='tight')
            plt.close()

    if save_npy:
        np.save(outdir / f"{prefix}_object_field.npy", u)
        np.save(outdir / f"{prefix}_fourier_field_centered.npy", Uc)
        np.save(outdir / f"{prefix}_errors.npy", errors)


# ============ 主函数（CLI） ============

def load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"仅支持二维数组，文件：{path}，实际维度：{arr.ndim}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Gerchberg–Saxton (GS) 相位恢复程序（含 ER 变体）")
    parser.add_argument("--method", choices=["gs", "er"], default="gs", help="选择算法：gs 或 er（Fienup Error-Reduction）")
    parser.add_argument("--fourier-mag", type=str, help="中心化的频域振幅 .npy 路径")
    parser.add_argument("--object-amp", type=str, help="物域振幅 .npy 路径（GS 需要）")
    parser.add_argument("--support", type=str, help="物域支持掩膜 .npy 路径（ER 可选）")
    parser.add_argument("--iters", type=int, default=200, help="迭代次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--positivity", action="store_true", help="ER 中启用非负约束（需 enforce-real 配合）")
    parser.add_argument("--enforce-real", action="store_true", help="ER 中启用实值约束（取实部）")
    parser.add_argument("--save-png", action="store_true", help="保存 PNG 可视化")
    parser.add_argument("--save-npy", action="store_true", help="保存 Numpy 结果")
    parser.add_argument("--outdir", type=str, default="results", help="输出目录")
    parser.add_argument("--demo", choices=["gs", "er"], help="运行内置演示（无需外部数据）")

    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.demo is not None:
        # 内置演示：生成合成目标
        amp = make_synthetic_object((256, 256))

        if args.demo == "gs":
            # 构造“真值”物域复场（随机真相位）并得到目标频域振幅
            rng = np.random.default_rng(args.seed)
            phi_true = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
            u_true = amp * np.exp(1j * phi_true)
            U_true = _fft2_centered(u_true)
            fourier_mag = amplitude(U_true)

            result = gerchberg_saxton(fourier_mag, amp, iters=args.iters, seed=args.seed)
            save_results(outdir, result, save_png=args.save_png or True, save_npy=args.save_npy,
                         prefix="demo_gs",
                         object_amp_for_display=amp,
                         fourier_mag_target=fourier_mag)

        elif args.demo == "er":
            # 仅频域振幅 + 支持；真值用于生成 fourier_mag
            rng = np.random.default_rng(args.seed)
            phi_true = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
            u_true = amp * np.exp(1j * phi_true)
            U_true = _fft2_centered(u_true)
            fourier_mag = amplitude(U_true)
            support = make_support_from_amp(amp, thresh=0.08)

            result = fienup_error_reduction(fourier_mag, support=support, iters=args.iters,
                                            seed=args.seed, positivity=args.positivity,
                                            enforce_real=args.enforce_real)
            save_results(outdir, result, save_png=args.save_png or True, save_npy=args.save_npy,
                         prefix="demo_er",
                         object_amp_for_display=amp,
                         fourier_mag_target=fourier_mag)

        print(f"[OK] 演示完成。结果已保存到：{outdir.resolve()}")
        return

    # 正常路径：从文件读取
    if args.fourier_mag is None:
        raise SystemExit("错误：--fourier-mag 必填（.npy 文件，中心化频域振幅）。")

    fourier_mag = load_npy(args.fourier_mag)

    if args.method == "gs":
        if args.object_amp is None:
            raise SystemExit("错误：GS 需提供 --object-amp（.npy 文件，物域振幅）。")
        object_amp = load_npy(args.object_amp)
        result = gerchberg_saxton(fourier_mag, object_amp, iters=args.iters, seed=args.seed)

        save_results(outdir, result, save_png=args.save_png, save_npy=args.save_npy,
                     prefix="gs",
                     object_amp_for_display=object_amp,
                     fourier_mag_target=fourier_mag)

    else:  # ER
        support = load_npy(args.support) if args.support is not None else None
        result = fienup_error_reduction(fourier_mag, support=support, iters=args.iters, seed=args.seed,
                                        positivity=args.positivity, enforce_real=args.enforce_real)

        save_results(outdir, result, save_png=args.save_png, save_npy=args.save_npy,
                     prefix="er",
                     object_amp_for_display=None,
                     fourier_mag_target=fourier_mag)

    print(f"[OK] 运行完成。结果已保存到：{outdir.resolve()}")


if __name__ == "__main__":
    main()
