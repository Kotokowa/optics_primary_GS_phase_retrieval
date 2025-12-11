import numpy as np

# 读取npy文件
data = np.load('temp_ratio/error_history.npy', allow_pickle=True)

'''
# 保存完整数据到文本文件
with open('my_npy.txt', 'w', encoding='utf-8') as f:
    f.write("完整相位数据 (50×50 复数数组)\n")
    f.write("=" * 60 + "\n\n")
    
    for i in range(300):
        f.write(f"行 {i:3d}: [")
        for j in range(300):
            if j > 0:
                f.write(", ")
            f.write(f"{data[i,j]: .4f}")
        f.write("]\n")

print("完整数据已保存到 my_npy.txt")

'''
with open('my_npy.txt', 'w', encoding='utf-8') as f:
    f.write("完整数据 (单列数组)\n")
    f.write("=" * 60 + "\n\n")
    
    for i in range(2000):
        f.write(f"行 {i:3d}: [")
        f.write(f"{data[i]: .4f}")
        f.write("]\n")

print("完整数据已保存到 my_npy.txt")
