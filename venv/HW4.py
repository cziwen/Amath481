import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

# 定义参数
L = 10
m = 8  # 在 x 和 y 方向的网格点数
n = m * m  # 矩阵的总大小
delta = (2 * L) / m  # 网格间距

# ===================== 构建 A 矩阵 ===========================

# 创建基本向量
e0 = np.zeros ((n, 1))  # 零向量
e1 = np.ones ((n, 1))  # 全为 1 的向量
e2 = np.copy (e1)  # 拷贝全 1 向量
e4 = np.copy (e0)  # 拷贝零向量

# 调整 e2 和 e4 的值，以便用于周期性边界条件
for j in range (1, m + 1):
    e2[m * j - 1] = 0  # 将每 m^th 值设为 0
    e4[m * j - 1] = 1  # 将每 m^th 值设为 1

# 为周期性边界条件调整 e3 和 e5
e3 = np.zeros_like (e2)
e5 = np.zeros_like (e4)

# 将 e2 和 e4 向量向左和向右分别移动一位给到e3 和 e5，形成周期性边界条件
for i in range (1, n):
    e3[i] = e2[i - 1]
    e5[i] = e4[i - 1]
e3[0] = e2[-1]
e5[0] = e4[-1]

# 放置对角元素来构建 A 矩阵
diagonals_A = [e1.flatten (), e1.flatten (), e5.flatten (),
               e2.flatten (), -4 * e1.flatten (), e3.flatten (),
               e4.flatten (), e1.flatten (), e1.flatten ()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]

A = spdiags (diagonals_A, offsets_A, n, n) / (delta ** 2)
A_hw4 = A

# 可视化 A 矩阵结构
# plt.figure (figsize=(8, 8))
# plt.spy (A)
# plt.title ('Matrix Structure A')
# plt.show ()

# ===================== 构建 B 矩阵 (∂x) ===========================

# B 矩阵的对角线设置需要包括周期性边界
diagonals_B = [e1.flatten(), -e1.flatten() , e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]  # 中心差分在 x 方向的偏移
B = spdiags (diagonals_B, offsets_B, n, n) / (2 * delta)
B_hw4 = B

# 可视化 B 矩阵结构
# plt.figure (figsize=(8, 8))
# plt.spy (B)
# plt.title ('Matrix Structure B (∂x)')
# plt.show ()

# ===================== 构建 C 矩阵 (∂y) ===========================

# 修改一下e4，向右边移动一格，然后附值给e1
for i in range (1, n):
    e1[i] = e4[i - 1]


# C 矩阵的对角线设置需要包括周期性边界
diagonals_C = [e1.flatten (), -e2.flatten (), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,  m - 1 ]  # 中心差分在 y 方向的偏移
C = spdiags (diagonals_C, offsets_C, n, n) / (2 * delta)
C_hw4 = C

# 可视化 C 矩阵结构
# plt.figure (figsize=(8, 8))
# plt.spy (C)
# plt.title ('Matrix Structure C (∂y)')
# plt.show ()


#================ 将矩阵 A, B, C 以密集格式输出 ==============
A_dense = A.toarray ()
B_dense = B.toarray ()
C_dense = C.toarray ()


A1 = A_dense
A2 = B_dense
A3 = C_dense

print ("Matrix A:\n", A_dense)
print ("\nMatrix B (∂x):\n", B_dense)
print ("\nMatrix C (∂y):\n", C_dense)
