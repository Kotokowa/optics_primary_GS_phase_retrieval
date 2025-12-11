# -*- coding: utf-8 -*-
"""
Gerchberg–Saxton (GS) 相位恢复（两平面幅度已知）最小示例
----------------------------------------------------------------
给定：
  - 物/输入平面幅度 A(x)    （N×N 实数非负）
  - 频域/傅里叶平面幅度 M(u)（N×N 实数非负）
目标：
  - 恢复物平面复场 g(x) 的“相位”（幅度约束固定为 A）

算法（基础 GS，其它先验/变体如 HIO、RAAR 不在此示例中）：
  1) g_k  --FFT-->  G_pred
  2) 用测得幅度替换： G_proj = M * exp(i * angle(G_pred))
  3) 反变换：         g_tmp  = IFFT(G_proj)
  4) 用测得幅度替换： g_{k+1}= A * exp(i * angle(g_tmp))
  5) 迭代并监控误差： E(k) = || |G_pred| - M ||_2 / ||M||_2

提示：
  - 该误差在“投影前”的 |G_pred| 与测量 M 比较；若在投影后比较会恒为 0。
  - FFT 采用“中心化 + 正交化”规范（方便数值稳定与直观对齐）。
  - GS 存在全局相位不定性：恢复的相位与真值差一个常数相位。
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# ===========
# 工具函数区
# ===========

def fft2c(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """
    中心化二维 FFT（先 ifftshift 再 fft2，再 fftshift）
    norm='ortho' 让正反变换单位能量，便于幅度一致性
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm=norm))

def ifft2c(X: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """中心化二维 IFFT（先 ifftshift 再 ifft2，再 fftshift）"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X), norm=norm))

def project_to_amplitude(Z: np.ndarray, target_amp: np.ndarray) -> np.ndarray:
    """
    将复场 Z 投影到“给定幅度 target_amp”的集合：
        P(Z) = target_amp * exp(i * angle(Z))
    说明：不进行除法，避免 |Z|≈0 时的数值问题；angle(0)=0 合理。
    """
    return target_amp * np.exp(1j * np.angle(Z))

def relative_fourier_amplitude_error(G_pred_abs: np.ndarray, M: np.ndarray, eps: float = 1e-12) -> float:
    """
    归一化频域幅度误差：
        E = || |G_pred| - M ||_2 / ||M||_2
    """
    num = np.linalg.norm(G_pred_abs - M)
    den = np.linalg.norm(M) + eps
    return float(num / den)

def align_phase_global(phi_est: np.ndarray, phi_true: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    计算“最优全局相位偏移” α，使得 e^{i(φ_est - φ_true - α)} 的整体相位误差最小。
    做法：α = angle( Σ e^{i(φ_est - φ_true)} )。
    返回 α（弧度）。
    """
    if mask is None:
        w = np.exp(1j * (phi_est - phi_true))
    else:
        w = np.exp(1j * (phi_est - phi_true)) * (mask.astype(float))
    alpha = np.angle(np.sum(w))
    return float(alpha)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """将相位差值折叠到 (-π, π]"""
    return (x + np.pi) % (2 * np.pi) - np.pi

# =========================
# GS 主函数（两平面幅度已知）
# =========================

def gerchberg_saxton(
    A: np.ndarray,
    M: np.ndarray,
    num_iters: int = 300,
    init_phase: Optional[np.ndarray] = None,
    fft_norm: str = "ortho",
    stop_tol: float = 1e-8,
    patience: int = 20,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Gerchberg–Saxton 相位恢复（最基础版，两个平面幅度约束）

    参数：
        A         : 物/输入平面幅度 (N×N，非负)
        M         : 频域（傅里叶面）幅度 (N×N，非负)
        num_iters : 最大迭代次数
        init_phase: 初始相位 (N×N, 弧度)。若 None 则使用均匀随机 [-π, π)
        fft_norm  : FFT 规范（默认 'ortho'）
        stop_tol  : 早停阈值（误差相对变化小于该值判定“收敛”）
        patience  : 连续多少次改进低于阈值则停止
        verbose   : 是否打印迭代日志

    返回：
        dict，包括：
            'g'      : 恢复的物平面复场（幅度已被强制为 A）
            'phase'  : 恢复相位（与 g 的 angle 一致）
            'errors' : 每次迭代的频域幅度误差列表
    """
    # ---- 输入检查 ----
    if A.shape != M.shape:
        raise ValueError("A 与 M 的形状必须一致（均为 N×N）。")
    if np.any(A < 0) or np.any(M < 0):
        raise ValueError("A 和 M 必须为非负实数阵列。")
    A = A.astype(np.float64, copy=False)
    M = M.astype(np.float64, copy=False)
    N = A.shape[0]

    # ---- 初始化复场 g0 = A * exp(i * φ0) ----
    if init_phase is None:
        rng = np.random.default_rng(2025)  # 固定随机种子，结果可复现
        phi0 = rng.uniform(-np.pi, np.pi, size=A.shape)
    else:
        if init_phase.shape != A.shape:
            raise ValueError("init_phase 形状必须与 A 相同。")
        phi0 = init_phase
    g = A * np.exp(1j * phi0)

    # ---- 迭代主循环 ----
    errors: list[float] = []
    no_improve = 0
    last_E = np.inf

    for k in range(1, num_iters + 1):
        # 1) 物 -> 频：预测频谱
        G_pred = fft2c(g, norm=fft_norm)

        # 计算误差（在频域投影前）
        E = relative_fourier_amplitude_error(np.abs(G_pred), M)
        errors.append(E)

        # 迭代信息与早停判断
        if verbose and (k == 1 or k % 50 == 0):
            print(f"[Iter {k:4d}]  E_F = {E:.6e}")

        if last_E - E < stop_tol:  # 改善不足
            no_improve += 1
        else:
            no_improve = 0
        last_E = E
        if no_improve >= patience:
            if verbose:
                print(f"早停：连续 {patience} 次改进 < {stop_tol:.1e}。")
            break

        # 2) 频域投影（替换幅度为测量 M，保留相位）
        G_proj = project_to_amplitude(G_pred, M)

        # 3) 频 -> 物
        g_tmp = ifft2c(G_proj, norm=fft_norm)

        # 4) 物域投影（替换幅度为 A，保留相位）
        g = project_to_amplitude(g_tmp, A)

    result = {
        "g": g,
        "phase": np.angle(g),
        "errors": np.asarray(errors, dtype=np.float64),
    }
    return result

# =========================
# 可运行示例（合成数据）
# =========================

def make_circular_aperture(N: int, radius_ratio: float = 0.7) -> np.ndarray:
    """
    生成半径为 radius_ratio * (N/2) 的圆孔幅度分布（值为 0/1）
    """
    yy, xx = np.mgrid[-N//2:N//2, -N//2:N//2]
    r = np.sqrt(xx**2 + yy**2)
    R = radius_ratio * (N / 2.0)
    A = (r <= R).astype(float)
    return A

def make_quadratic_phase(N: int, strength: float = 2.0) -> np.ndarray:
    """
    生成类“离焦”的二次相位：φ(x,y) = strength * π * (x^2 + y^2) / (R^2)
    强度 strength 控制曲率；结果范围约为 [-π*strength, π*strength]。
    """
    yy, xx = np.mgrid[-N//2:N//2, -N//2:N//2]
    R = (N / 2.0)
    phi = strength * np.pi * (xx**2 + yy**2) / (R**2)
    # 将相位折叠到 (-π, π]，便于可视化
    return wrap_to_pi(phi)

def demo():
    # --------- 1) 构造“真值” ---------
    N = 256
    A_true = make_circular_aperture(N, radius_ratio=0.7)     # 物面幅度 A
    phi_true = make_quadratic_phase(N, strength=1.5)         # 真值相位（未知）
    g_true = A_true * np.exp(1j * phi_true)                  # 物面复场（真值）
    M_meas = np.abs(fft2c(g_true, norm="ortho"))             # 频域幅度 M（测量）

    # --------- 2) 用 GS 重建 ---------
    out = gerchberg_saxton(
        A=A_true,
        M=M_meas,
        num_iters=500,
        init_phase=None,        # 随机初相
        fft_norm="ortho",
        stop_tol=1e-8,
        patience=30,
        verbose=True
    )
    g_est = out["g"]
    phi_est = out["phase"]
    errors = out["errors"]

    # --------- 3) 评估（考虑全局相位不定性） ---------
    mask = A_true > 0.5
    alpha = align_phase_global(phi_est, phi_true, mask=mask)  # 最优全局相位偏移
    phase_diff_wrapped = wrap_to_pi(phi_est - (phi_true + alpha))
    rmse = np.sqrt(np.mean((phase_diff_wrapped[mask])**2))

    print(f"\n最终归一化频域幅度误差  E_F = {errors[-1]:.6e}")
    print(f"相位 RMSE（去全局相位，单位弧度）= {rmse:.4f}")

    # --------- 4) 可视化 ---------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.ravel()

    im0 = ax[0].imshow(A_true, cmap="gray")
    ax[0].set_title("物面幅度 A（已知）"); plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(np.log1p(M_meas), cmap="gray")
    ax[1].set_title("频域幅度 M 的对数显示（测量）"); plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(phi_true, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("真值相位 φ_true"); plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(np.angle(g_est), cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax[3].set_title("恢复相位 φ_est（未对齐）"); plt.colorbar(im3, ax=ax[3], fraction=0.046)

    im4 = ax[4].imshow(phase_diff_wrapped, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax[4].set_title("相位残差（去全局相位）"); plt.colorbar(im4, ax=ax[4], fraction=0.046)

    ax[5].plot(errors, lw=2)
    ax[5].set_xlabel("Iteration")
    ax[5].set_ylabel("E_F (Fourier amplitude error)")
    ax[5].set_title("误差收敛曲线")

    for a in ax: a.set_xticks([]); a.set_yticks([])
    ax[5].grid(True); ax[5].set_xticks(np.linspace(0, len(errors), 6))
    plt.tight_layout()
    plt.show()

# 直接运行本脚本时，执行示例
if __name__ == "__main__":
    demo()
