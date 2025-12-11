import numpy as np
import importlib.util

# 动态加载脚本为模块
spec = importlib.util.spec_from_file_location("gspr", "gs_phase_retrieval.py")
gspr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gspr)

# 读取你的数据（二维 numpy 数组）
fourier_mag = np.load("fourier_mag.npy")   # 中心化的 |U(f)|
object_amp  = np.load("object_amp.npy")    # |u(x)|，仅 GS 用

# 运行 GS
res = gspr.gerchberg_saxton(fourier_mag, object_amp, iters=300, seed=0)
u  = res.field_object           # 物域复场（complex64/128）
ph = np.angle(u)                # 相位
np.save("recovered_field.npy", u)

#可选ER
'''
support = np.load("support.npy")     # 0/1 掩膜
res = gspr.fienup_error_reduction(fourier_mag, support=support,
                                  iters=1000, seed=0,
                                  positivity=True, enforce_real=True)
u = res.field_object
'''

'''
powershell:
python .\gs_phase_retrieval.py `
  --method gs `
  --fourier-mag .\HEART_384x384_fourier_mag.npy `
  --object-amp .\HEART_384x384_object_amp.npy `
  --iters 500 --seed 0 --outdir .\results_STAR --save-png --save-npy
'''