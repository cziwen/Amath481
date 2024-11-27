import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import lu, solve_triangular
import time
from scipy.sparse.linalg import bicgstab, gmres


# Ziwen Chen

# 定义参数
L = 10
m = 64  # 在 x 和 y 方向的网格点数
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

# 可视化 A 矩阵结构
# plt.figure (figsize=(8, 8))
# plt.spy (A)
# plt.title ('Matrix Structure A')
# plt.show ()

# ===================== 构建 B 矩阵 (∂x) ===========================

# B 矩阵的对角线设置需要包括周期性边界
diagonals_B = [e1.flatten (), -e1.flatten (), e1.flatten (), -e1.flatten ()]
offsets_B = [-(n - m), -m, m, (n - m)]  # 中心差分在 x 方向的偏移
B = spdiags (diagonals_B, offsets_B, n, n) / (2 * delta)

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
diagonals_C = [e1.flatten (), -e2.flatten (), e3.flatten (), -e4.flatten ()]
offsets_C = [-m + 1, -1, 1, m - 1]  # 中心差分在 y 方向的偏移
C = spdiags (diagonals_C, offsets_C, n, n) / (2 * delta)

# 可视化 C 矩阵结构
# plt.figure (figsize=(8, 8))
# plt.spy (C)
# plt.title ('Matrix Structure C (∂y)')
# plt.show ()


# ================ 将矩阵 A, B, C 以密集格式输出 ==============
A_dense = A.toarray ()
A_dense[0, 0] = 2 / (delta ** 2)
B_dense = B.toarray ()
C_dense = C.toarray ()

# print ("Matrix A:\n", A_dense.shape)
# print ("\nMatrix B (∂x):\n", B_dense.shape)
# print ("\nMatrix C (∂y):\n", C_dense.shape)

# =============================== 定义参数 =================================
# 定义参数
tspan = np.arange (0, 4.5, 0.5)  # 时间范围，步长为0.5
nu = 0.001  # 粘性系数
Lx, Ly = 20, 20  # 空间范围
nx, ny = 64, 64  # 网格点数
N = nx * ny

# 定义空间域和初始条件
x2 = np.linspace (-Lx / 2, Lx / 2, nx + 1)
x = x2[:nx]
y2 = np.linspace (-Ly / 2, Ly / 2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid (x, y)
w0 = np.exp (-X ** 2 - Y ** 2 / 20).flatten ()  # 初始漩涡场，同时转成一维向量

# 定义谱域k值
kx = (2 * np.pi / Lx) * np.concatenate ((np.arange (0, nx / 2), np.arange (-nx / 2, 0)))
kx[0] = 1e-6  # 避免零除
ky = (2 * np.pi / Ly) * np.concatenate ((np.arange (0, ny / 2), np.arange (-ny / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid (kx, ky)
K = KX ** 2 + KY ** 2

# ================================= FFT ======================================

# 定义ODE系统的右端项
def spc_rhs (t, wt2, nx, ny, N, KX, KY, K, nu):
    # fft转换 omega
    w = wt2.reshape ((nx, ny))
    wt = fft2 (w)

    # 求解psi
    psit = - wt / K
    psi = np.real (ifft2 (psit)).reshape (N)

    rhs = (
            nu * A_dense.dot (wt2)  # nu (A w)
            + (B_dense.dot (wt2)) * (C_dense.dot (psi))  # - (B w) * (C psi) 和jenny答案符号相反（已经修改)
            - (B_dense.dot (psi)) * (C_dense.dot (wt2))  # + (B psi) * (C w)
    )

    return rhs


start_time = time.time ()  # 记录时间

# 使用 solve_ivp 进行求解
sol = solve_ivp (
    spc_rhs,
    (tspan[0], tspan[-1]),
    w0,
    t_eval=tspan,
    args=(nx, ny, N, KX, KY, K, nu),
    method='RK45'
)

# 记录时间
end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for FFT: {elapsed_time:.2f} seconds")

# 提取解
wtsol_fft = sol.y  # 转置，保持与 tspan 对应
# 保存最终解 A1
A1 = wtsol_fft
print("A1", A1)


# # ================= 可视化结果 FFT ===================
# sol_to_plot = A1
#
# # 确定每列的数据可以重塑为 n x n 的矩阵
# n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field - FFT')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')
#
#
# def update (frame):
#     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
# anim.save ('/Users/ziwenchen/PycharmProjects/Amath481/vorticity_evolution_FFT.gif', writer='imagemagick', fps=2)


# ============================== A/b ================================

def ab_rhs (t, w, A_dense, B_dense, C_dense, nu):
    # 使用矩阵操作来计算右端项
    psi = np.linalg.solve (A_dense, w)  # 解 A ψ = w，得到 ψ

    rhs = (
            nu * A_dense.dot (w)  # nu * A * w
            + (B_dense.dot (w)) * (C_dense.dot (psi))  # + (B w) * (C ψ)
            - (B_dense.dot (psi)) * (C_dense.dot (w))  # - (B ψ) * (C w)
    )
    return rhs


start_time = time.time ()  # 记录时间

# 使用 solve_ivp
sol_ab = solve_ivp (
    ab_rhs,
    (tspan[0], tspan[-1]),  # 时间范围
    w0,  # 初始条件
    t_eval=tspan,  # 时间步
    args=(A_dense, B_dense, C_dense, nu),  # 参数
    method='RK45'  # 数值求解方法
)

# 记录时间
end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for A/b: {elapsed_time:.2f} seconds")

# 提取解
wtsol_ab = sol_ab.y  # 转置，保持与 tspan 对应
A2 = wtsol_ab
print("A2", A2)

# # ================= 可视化结果 A/b ===================
# sol_to_plot = A2
#
# # 确定每列的数据可以重塑为 n x n 的矩阵
# n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field - A/b')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')
#
#
# def update (frame):
#     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
# anim.save ("/Users/ziwenchen/PycharmProjects/Amath481/vorticity_evolution_Ab.gif", writer='imagemagick', fps=2)

# ============================== LU ================================

start_time = time.time ()  # 记录时间

P, L, U = lu (A_dense)


def lu_rhs (t, w, A_dense, B_dense, C_dense, nu, L, U, P):
    Pw = np.dot (P, w)
    y = solve_triangular (L, Pw, lower=True)
    psi = solve_triangular (U, y, lower=False)

    rhs = (
            nu * A_dense.dot (w)  # nu * A * w
            + (B_dense.dot (w)) * (C_dense.dot (psi))  # + (B w) * (C ψ)
            - (B_dense.dot (psi)) * (C_dense.dot (w))  # - (B ψ) * (C w)
    )

    return rhs


sol = solve_ivp (
    lu_rhs,
    (tspan[0], tspan[-1]),
    w0,
    t_eval=tspan,
    args=(A_dense, B_dense, C_dense, nu, L, U, P),
    method='RK45'
)

end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for LU: {elapsed_time:.2f} seconds")

wtsol_lu = sol.y
A3 = wtsol_lu
print ("A3", A3)

# # # ================= 可视化结果 LU ===================
# sol_to_plot = A3
#
# # 确定每列的数据可以重塑为 n x n 的矩阵
# n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field - A/b')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')
#
#
# def update (frame):
#     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
# anim.save ("/Users/ziwenchen/PycharmProjects/Amath481/vorticity_evolution_LU.gif", writer='imagemagick', fps=2)

# ============================== BICGSTAB ================================

# 将修改过A[0,0] = 2的 A_dense 转换成 A_sparse
A_sparse = csr_matrix (A_dense)


def bicgstab_rhs (t, w, A_dense, B_dense, C_dense, nu):
    psi, info = bicgstab (A_sparse, w, atol=1e-8, maxiter=1000)

    rhs = (
            nu * A_dense.dot (w)  # nu * A * w
            + (B_dense.dot (w)) * (C_dense.dot (psi))  # + (B w) * (C ψ)
            - (B_dense.dot (psi)) * (C_dense.dot (w))  # - (B ψ) * (C w)
    )

    return rhs


start_time = time.time ()  # 记录时间

sol = solve_ivp (
    bicgstab_rhs,
    (tspan[0], tspan[-1]),
    w0,
    t_eval=tspan,
    args=(A_dense, B_dense, C_dense, nu),
    method='RK45'
)

end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for BICGSTAB: {elapsed_time:.2f} seconds")

wtsol_bicgstab = sol.y
A4 = wtsol_bicgstab
print ("A4", A4)

# # ================= 可视化结果 BICGSTAB ===================
# sol_to_plot = A4
#
# # 确定每列的数据可以重塑为 n x n 的矩阵
# n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field - BICGSTAB')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')
#
#
# def update (frame):
#     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
# anim.save("/Users/ziwenchen/PycharmProjects/Amath481/vorticity_evolution_LU.gif", writer='pillow')

# ============================ GMRES ==============================

# 将修改过A[0,0] = 2的 A_dense 转换成 A_sparse
A_sparse = csr_matrix (A_dense)


def gmres_rhs(t, w, A_dense, B_dense, C_dense, nu):
    psi, info = gmres (A_sparse, w, atol=1e-8, restart=50, maxiter=1000)

    rhs = (
            nu * A_dense.dot (w)  # nu * A * w
            + (B_dense.dot (w)) * (C_dense.dot (psi))  # + (B w) * (C ψ)
            - (B_dense.dot (psi)) * (C_dense.dot (w))  # - (B ψ) * (C w)
    )

    return rhs


start_time = time.time ()

sol = solve_ivp (
    gmres_rhs,
    (tspan[0], tspan[-1]),
    w0,
    t_eval=tspan,
    args=(A_dense, B_dense, C_dense, nu),
    method='RK45'
)

end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for GMRES: {elapsed_time:.2f} seconds")

wtsol_gmres = sol.y
A5 = wtsol_gmres
print("A5", A5)

# ================= 可视化结果 BICGSTAB ===================
# sol_to_plot = A5
#
# # 确定每列的数据可以重塑为 n x n 的矩阵
# n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field - GMRES')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')
#
#
# def update (frame):
#     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
# anim.save("/Users/ziwenchen/PycharmProjects/Amath481/vorticity_evolution_GMRES.gif", writer='pillow')
# ===========================================================

# ================================================== BELOW ARE CODES TO DRAW GIF FOR 5 METHODS OF SOLVING PSI ====================================================
# ============== MUTE AS DEFAULT==================
# import time
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.fft import fft2, ifft2
# from scipy.integrate import solve_ivp
# from scipy.sparse import spdiags
#
# # 定义参数
# L = 10
# m = 64  # 在 x 和 y 方向的网格点数
# n = m * m  # 矩阵的总大小
# delta = (2 * L) / m  # 网格间距
#
# # ===================== 构建 A 矩阵 ===========================
#
# # 创建基本向量
# e0 = np.zeros ((n, 1))  # 零向量
# e1 = np.ones ((n, 1))  # 全为 1 的向量
# e2 = np.copy (e1)  # 拷贝全 1 向量
# e4 = np.copy (e0)  # 拷贝零向量
#
# # 调整 e2 和 e4 的值，以便用于周期性边界条件
# for j in range (1, m + 1):
#     e2[m * j - 1] = 0  # 将每 m^th 值设为 0
#     e4[m * j - 1] = 1  # 将每 m^th 值设为 1
#
# # 为周期性边界条件调整 e3 和 e5
# e3 = np.zeros_like (e2)
# e5 = np.zeros_like (e4)
#
# # 将 e2 和 e4 向量向左和向右分别移动一位给到e3 和 e5，形成周期性边界条件
# for i in range (1, n):
#     e3[i] = e2[i - 1]
#     e5[i] = e4[i - 1]
# e3[0] = e2[-1]
# e5[0] = e4[-1]
#
# # 放置对角元素来构建 A 矩阵
# diagonals_A = [e1.flatten (), e1.flatten (), e5.flatten (),
#                e2.flatten (), -4 * e1.flatten (), e3.flatten (),
#                e4.flatten (), e1.flatten (), e1.flatten ()]
# offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
#
# A = spdiags (diagonals_A, offsets_A, n, n) / (delta ** 2)
#
# # 可视化 A 矩阵结构
# # plt.figure (figsize=(8, 8))
# # plt.spy (A)
# # plt.title ('Matrix Structure A')
# # plt.show ()
#
# # ===================== 构建 B 矩阵 (∂x) ===========================
#
# # B 矩阵的对角线设置需要包括周期性边界
# diagonals_B = [e1.flatten (), -e1.flatten (), e1.flatten (), -e1.flatten ()]
# offsets_B = [-(n - m), -m, m, (n - m)]  # 中心差分在 x 方向的偏移
# B = spdiags (diagonals_B, offsets_B, n, n) / (2 * delta)
#
# # 可视化 B 矩阵结构
# # plt.figure (figsize=(8, 8))
# # plt.spy (B)
# # plt.title ('Matrix Structure B (∂x)')
# # plt.show ()
#
# # ===================== 构建 C 矩阵 (∂y) ===========================
#
# # 修改一下e4，向右边移动一格，然后附值给e1
# for i in range (1, n):
#     e1[i] = e4[i - 1]
#
# # C 矩阵的对角线设置需要包括周期性边界
# diagonals_C = [e1.flatten (), -e2.flatten (), e3.flatten (), -e4.flatten ()]
# offsets_C = [-m + 1, -1, 1, m - 1]  # 中心差分在 y 方向的偏移
# C = spdiags (diagonals_C, offsets_C, n, n) / (2 * delta)
#
# # 可视化 C 矩阵结构
# # plt.figure (figsize=(8, 8))
# # plt.spy (C)
# # plt.title ('Matrix Structure C (∂y)')
# # plt.show ()
#
#
# # ================ 将矩阵 A, B, C 以密集格式输出 ==============
# A_dense = A.toarray ()
# A_dense[0, 0] = 2 / (delta ** 2)
# B_dense = B.toarray ()
# C_dense = C.toarray ()
#
# # print ("Matrix A:\n", A_dense.shape)
# # print ("\nMatrix B (∂x):\n", B_dense.shape)
# # print ("\nMatrix C (∂y):\n", C_dense.shape)
#
# # =============================== 定义参数 =================================
# # 定义参数
# tspan = np.arange (0, 40, 0.5)  # 时间范围，步长为0.5
# nu = 0.001  # 粘性系数
# Lx, Ly = 20, 20  # 空间范围
# nx, ny = 64, 64  # 网格点数
# N = nx * ny
#
# # 定义空间域和初始条件
# x2 = np.linspace (-Lx / 2, Lx / 2, nx + 1)
# x = x2[:nx]
# y2 = np.linspace (-Ly / 2, Ly / 2, ny + 1)
# y = y2[:ny]
# X, Y = np.meshgrid (x, y)
#
# # 定义一系列 初始漩涡场
# w0s = []
#
# # Two Oppositely "charged" Gaussian vorticies
# # 定义omega参数
# sigma = 1.0  # 涡旋宽度
# A1, A2 = 5.0, -5.0  # 两个涡旋的振幅，分别为正负
# x0_1, y0_1 = -2.0, 0.0  # 第一个涡旋的中心
# x0_2, y0_2 = 2.0, 0.0  # 第二个涡旋的中心
#
# # 计算每个网格点的涡度值
# w0_2d = (
#         A1 * np.exp (-((X - x0_1) ** 2 + (Y - y0_1) ** 2) / sigma ** 2) +
#         A2 * np.exp (-((X - x0_2) ** 2 + (Y - y0_2) ** 2) / sigma ** 2)
# )
#
# w0s.append (w0_2d.flatten ())  # 添加到漩涡场list
#
# # Two same "charged" Gaussian vorticies
# # 定义omega参数
# sigma = 1.0  # 涡旋宽度
# A1, A2 = 5.0, 5.0  # 两个涡旋的振幅，分别为正负
# x0_1, y0_1 = -2.0, 0.0  # 第一个涡旋的中心
# x0_2, y0_2 = 2.0, 0.0  # 第二个涡旋的中心
#
# # 计算每个网格点的涡度值
# w0_2d = (
#         A1 * np.exp (-((X - x0_1) ** 2 + (Y - y0_1) ** 2) / sigma ** 2) +
#         A2 * np.exp (-((X - x0_2) ** 2 + (Y - y0_2) ** 2) / sigma ** 2)
# )
#
# w0s.append (w0_2d.flatten ())  # 添加到漩涡场list
#
# # Two pairs of oppositely "charged" Gaussian vorticies
# # 定义omega参数
# sigma = 1.0
# A1, A2 = 5.0, -5.0  # amplitude
# A3, A4 = 5.0, -5.0  # amplitude
# x0_1, y0_1 = -3.0, -1.0  # 1st omega position
# x0_2, y0_2 = -3.0, 1.0  # 2nd omega position
# x0_3, y0_3 = 3.0, -1.0  # 3rd...
# x0_4, y0_4 = 3.0, 1.0  # ....
#
# w0_2d = (
#         A1 * np.exp (-((X - x0_1) ** 2 + (Y - y0_1) ** 2) / sigma ** 2) +  # 第一对
#         A2 * np.exp (-((X - x0_2) ** 2 + (Y - y0_2) ** 2) / sigma ** 2) +  # 第一对
#         A3 * np.exp (-((X - x0_3) ** 2 + (Y - y0_3) ** 2) / sigma ** 2) +  # 第二对
#         A4 * np.exp (-((X - x0_4) ** 2 + (Y - y0_4) ** 2) / sigma ** 2)  # 第二对
# )
#
# w0s.append (w0_2d.flatten ())
#
# # A random assortment(in posiion, strength, charge, ellipticity)
# # 随机生成多个高斯涡旋
# num_vortices = 15  # 涡旋数量
# sigma = 1.0
# w0_2d = np.zeros_like (X)
#
# for _ in range (num_vortices):
#     A = np.random.uniform (-5, 5)  # 随机振幅（正负随机）
#     x0 = np.random.uniform (-Lx / 2, Lx / 2)  # 随机位置
#     y0 = np.random.uniform (-Ly / 2, Ly / 2)
#     w0_2d += A * np.exp (-((X - x0) ** 2 + (Y - y0) ** 2) / sigma ** 2)
#
# w0s.append (w0_2d.flatten ())
#
# # 定义谱域k值
# kx = (2 * np.pi / Lx) * np.concatenate ((np.arange (0, nx / 2), np.arange (-nx / 2, 0)))
# kx[0] = 1e-6  # 避免零除
# ky = (2 * np.pi / Ly) * np.concatenate ((np.arange (0, ny / 2), np.arange (-ny / 2, 0)))
# ky[0] = 1e-6
# KX, KY = np.meshgrid (kx, ky)
# K = KX ** 2 + KY ** 2
#
#
# # ====================== FFT 求解 ===============
#
# # 定义ODE系统的右端项
# def spc_rhs (t, wt2, nx, ny, N, KX, KY, K, nu):
#     # fft转换 omega
#     w = wt2.reshape ((nx, ny))
#     wt = fft2 (w)
#
#     # 求解psi
#     psit = - wt / K
#     psi = np.real (ifft2 (psit)).reshape (N)
#
#     rhs = (
#             nu * A_dense.dot (wt2)  # nu (A w)
#             + (B_dense.dot (wt2)) * (C_dense.dot (psi))  # - (B w) * (C psi) 和jenny答案符号相反（已经修改)
#             - (B_dense.dot (psi)) * (C_dense.dot (wt2))  # + (B psi) * (C w)
#     )
#
#     return rhs
#
#
# w0s_name = ["Two Oppositely 'charged' Gaussian vorticies",
#             "Two same 'charged' Gaussian vorticies",
#             "Two pairs of oppositely 'charged' Gaussian vorticies",
#             "A random assortment(in posiion, strength, charge, ellipticity)"]
# results = []
# i = 0
# for w0 in w0s:
#     start_time = time.time ()  # 记录时间
#
#     # 使用 solve_ivp 进行求解
#     sol = solve_ivp (
#         spc_rhs,
#         (tspan[0], tspan[-1]),
#         w0,
#         t_eval=tspan,
#         args=(nx, ny, N, KX, KY, K, nu),
#         method='RK45'
#     )
#
#     # 记录时间
#     end_time = time.time ()
#     elapsed_time = end_time - start_time
#     print (f"Elapsed time for FFT: {elapsed_time:.2f} seconds")
#
#     # 提取解
#     wtsol_fft = sol.y  # 转置，保持与 tspan 对应
#     results.append (wtsol_fft)
#
#     # =============================可视化=============================
#     sol_to_plot = wtsol_fft
#
#     # 确定每列的数据可以重塑为 n x n 的矩阵
#     n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
#
#     fig, ax = plt.subplots (figsize=(6, 6))
#     cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
#     fig.colorbar (cax, ax=ax, label='Vorticity')
#     ax.set_title ('Vorticity Field - GMRES')
#     ax.set_xlabel ('x')
#     ax.set_ylabel ('y')
#
#     # 添加全局标题
#     fig.suptitle (f"{w0s_name[i]}", fontsize=16)  # 在正上方添加标题
#
#
#     def update (frame):
#         ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
#         cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
#         return cax,
#
#
#     anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
#     anim.save (f"/Users/ziwenchen/PycharmProjects/Amath481/Movie For Gaussian Vorticity/vorticity_evolution_{w0s_name[i]}.gif", writer='pillow')
#
#
#     i += 1 # Increment index for name
#
#
#
# # ================= 统一绘制结果 ==================
# # 转换为 NumPy 数组：results -> shape (len(w0s), N, len(tspan))
# results = np.array (results)
#
# # 确定网格大小
# n = int (np.sqrt (results.shape[1]))  # n = sqrt(N) = 64
#
# # 绘制合并动画
# fig, ax = plt.subplots (figsize=(6, 6))
# cax = ax.imshow (results[0, :, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
# fig.colorbar (cax, ax=ax, label='Vorticity')
# ax.set_title ('Vorticity Field')
#
# # 添加全局标题
# fig.suptitle ("All Combined together", fontsize=16)  # 在正上方添加标题
#
#
# def update (frame):
#     combined_field = np.zeros ((n, n))
#     for i in range (len (w0s)):
#         combined_field += results[i, :, frame].reshape ((n, n))  # 累加所有涡旋的演化结果
#     cax.set_data (combined_field)
#     ax.set_title (f'Combined Vorticity Field at t = {frame * 0.5:.2f}')
#     return cax,
#
#
# anim = FuncAnimation (fig, update, frames=results.shape[2], blit=True)
# anim.save ('/Users/ziwenchen/PycharmProjects/Amath481/Movie For Gaussian Vorticity/combined_vorticity_evolution.gif', writer='pillow', fps=2)
#
#


