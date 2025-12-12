# GS 相位恢复程序

### 功能

- 对于任意强度和相位的实空间光矢量得到的实空间强度和频域空间强度进行相位恢复。
- Robust for 噪声、各种长宽比。
- 已加入自动中心化功能和相位平移功能。
- 可以把得到的相位结果化成图片并附带 colorbar，标尺可自定义。
- 对于模拟测试，包含了制作测试集的完整程序。
- **NEW** 支持 Fresnel 和 FFT 两种算法，测试集也已经上传

### 用法

- 对于 FFT 算法，需要下载
  \- Published_12_06_FFT
        \- make_object_info_with_shapefunction.py
        \- make_gs_testset.py
        \- gs_phase_retrieval.py
  对于 Fresnel 算法，需要下载
  \- Published_12_12_Fresnel\former_v4
        \- make_object_cli.py
        \- make_dataset_cli.py
        \- gs_fresnel_misell.py
- 用法见 usage.md 文件。
- 含有 read_npy.py 文件，可用于把 npy 文件转成 txt 便于阅读。
