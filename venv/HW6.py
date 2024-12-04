import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

# def tanh (x):
#     return np.sinh (x) / np.cosh (x)
#
#
# # 定义参数
# L = 20  # 空间范围 [-L/2, L/2]
# n = 64  # 网格点数
# beta = 1
# D1, D2 = 0.1, 0.1
# t_span = np.arange (0, 4.5, 0.5)  # 时间范围
#
# # 创建空间网格
# x = np.linspace (-L / 2, L / 2, n, endpoint=False)
# y = np.linspace (-L / 2, L / 2, n, endpoint=False)
# dx = L / n
# dy = L / n
# X, Y = np.meshgrid (x, y)
#
# # 初始条件
# m = 1  # 螺旋数
# angle = np.angle (X + 1j * Y)
# radius = np.sqrt (X ** 2 + Y ** 2)
# U0 = tanh (radius) * np.cos (m * angle - radius)
# V0 = tanh (radius) * np.sin (m * angle - radius)
#
#
# # 反应项
# def lambda_A (U, V):
#     A2 = np.abs (U) ** 2 + np.abs (V) ** 2
#     return 1 - A2
#
#
# def omega_A (U, V):
#     A2 = np.abs (U) ** 2 + np.abs (V) ** 2
#     return -beta * A2
#
#
# # 定义方程右侧
# def rhs (t, Z):
#     # 从 Z 中提取 U 和 V 的频域表示
#     U_hat = Z[:n ** 2].reshape ((n, n))
#     V_hat = Z[n ** 2:].reshape ((n, n))
#
#     # 从频域转换回时域
#     U = ifft2 (U_hat)
#     V = ifft2 (V_hat)
#
#     # 计算反应项
#     dUdt_reaction = lambda_A (U, V) * U - omega_A (U, V) * V
#     dVdt_reaction = omega_A (U, V) * U + lambda_A (U, V) * V
#
#     # 转换回频域
#     dUdt_reaction_hat = fft2 (dUdt_reaction)
#     dVdt_reaction_hat = fft2 (dVdt_reaction)
#
#     # 计算拉普拉斯项
#     kx = 2 * np.pi * np.fft.fftfreq (U.shape[1], d=dx)
#     ky = 2 * np.pi * np.fft.fftfreq (U.shape[0], d=dy)
#     KX, KY = np.meshgrid (kx, ky)
#     laplacian_operator = -(KX ** 2 + KY ** 2)
#
#     dUdt_diffusion_hat = D1 * laplacian_operator * U_hat
#     dVdt_diffusion_hat = D2 * laplacian_operator * V_hat
#
#     # 总变化率
#     dUdt_hat = dUdt_reaction_hat + dUdt_diffusion_hat
#     dVdt_hat = dVdt_reaction_hat + dVdt_diffusion_hat
#
#     # 拼接结果
#     rhs = np.concatenate ([dUdt_hat.flatten (), dVdt_hat.flatten ()])
#     return rhs
#
#
# # 初始化（保留频域表示）
# Z0 = np.concatenate ([fft2 (U0).flatten (), fft2 (V0).flatten ()])
#
# # 数值求解
# sol = solve_ivp (rhs,
#                  (t_span[0], t_span[-1]),
#                  Z0,
#                  t_eval=t_span,
#                  method='RK45')
#
# # 输出解的形状和解
# print ("Solution shape:", sol.y.shape)
# print ("Solution array:", sol.y)
#
# A1 = sol.y
#
#
# # 可视化
#
#
# # 提取时域解并生成带时间标签的图片帧
#
#
# def generate_frames_with_labels (sol, n, t_span):
#     frames = []
#     U_sol = sol.y[:n ** 2, :].reshape ((n, n, -1))  # 提取频域解的 U 部分
#     V_sol = sol.y[n ** 2:, :].reshape ((n, n, -1))  # 提取频域解的 V 部分
#
#     for t_idx, t in enumerate (t_span):
#         # 提取当前时间步的 U 和 V
#         U_hat = U_sol[:, :, t_idx]
#         V_hat = V_sol[:, :, t_idx]
#         U = np.abs (ifft2 (U_hat))  # 转换为时域解并取幅值
#         V = np.abs (ifft2 (V_hat))  # 转换为时域解并取幅值
#
#         # 创建并列图像
#         fig, axes = plt.subplots (1, 2, figsize=(12, 6))
#
#         # 绘制 U 图像
#         cax1 = axes[0].imshow (U, extent=(-L / 2, L / 2, -L / 2, L / 2), cmap='viridis')
#         axes[0].set_title ("U (Amplitude)")
#         axes[0].set_xlabel ("x")
#         axes[0].set_ylabel ("y")
#         fig.colorbar (cax1, ax=axes[0], orientation='vertical', label="Amplitude")
#
#         # 绘制 V 图像
#         cax2 = axes[1].imshow (V, extent=(-L / 2, L / 2, -L / 2, L / 2), cmap='plasma')
#         axes[1].set_title ("V (Amplitude)")
#         axes[1].set_xlabel ("x")
#         axes[1].set_ylabel ("y")
#         fig.colorbar (cax2, ax=axes[1], orientation='vertical', label="Amplitude")
#
#         # 添加时间标签
#         fig.suptitle (f"Reaction-Diffusion System - Time: {t:.2f}", fontsize=16)
#
#         # 转换图像为 Pillow 格式
#         fig.canvas.draw ()
#         img = Image.fromarray (np.array (fig.canvas.renderer.buffer_rgba ()))
#         frames.append (img)
#         plt.close (fig)
#
#     return frames
#
#
# # 调用生成帧的函数
# frames = generate_frames_with_labels (sol, n, t_span)
#
# # 保存为 GIF
# gif_filename = "reaction_diffusion_uv_fft.gif"
# frames[0].save (
#     gif_filename,
#     save_all=True,
#     append_images=frames[1:],
#     duration=200,  # 每帧显示的时间 (毫秒)
#     loop=0  # 循环次数 (0 表示无限循环)
# )
#
# print (f"GIF saved as {gif_filename}")
#

# ============================================== Chebychev ===============================================

from numpy import *


# Chebyshev differentiation matrix
def cheb (N):
    if N == 0:
        D = 0.;
        x = 1.
    else:
        n = np.arange (0, N + 1)
        x = np.cos (np.pi * n / N).reshape (N + 1, 1)
        c = (np.hstack (([2.], np.ones (N - 1), [2.])) * (-1) ** n).reshape (N + 1, 1)
        X = np.tile (x, (1, N + 1))
        dX = X - X.T
        D = np.dot (c, 1. / c.T) / (dX + np.eye (N + 1))
        D -= np.diag (sum (D.T, axis=0))
    return D, x.reshape (N + 1)


# Generate Chebyshev grid and differentiation matrix

D1 = D2 = 0.1
beta = 1
L = 20

n = 30
n2 = (n + 1) ** 2
D, x = cheb (n)
D[n, :] = 0
D[0, :] = 0
Dxx = np.dot (D, D) / ((L / 2) ** 2)
y = x

I = np.eye (len (Dxx))
L = kron (I, Dxx) + kron (Dxx, I)  # 2D Laplacian

X, Y = np.meshgrid (x, y)
X = X * 10
Y = Y * 10

t_span = np.arange (0, 4.5, 0.5)  # 时间范围

# Define the spiral initial condition
m = 1  # Number of spirals
angle = np.angle (X + 1j * Y)
radius = np.sqrt (X ** 2 + Y ** 2)
U0 = np.tanh (radius) * np.cos (m * angle - radius)
V0 = np.tanh (radius) * np.sin (m * angle - radius)


# Reaction terms
def lambda_A (U, V):
    A2 = U ** 2 + V ** 2
    return 1 - A2


def omega_A (U, V):
    A2 = U ** 2 + V ** 2
    return -beta * A2


# 定义 rhs
def rhs (t, uv_t):
    n_rhs = n + 1

    # 提取 U 和 V
    ut, vt = uv_t[:n_rhs ** 2], uv_t[n_rhs ** 2:]

    # 反应项
    dUdt = (lambda_A (ut, vt) * ut - omega_A (ut, vt) * vt) + D1 * (L @ ut)
    dVdt = (omega_A (ut, vt) * ut + lambda_A (ut, vt) * vt) + D2 * (L @ vt)

    # 将结果展平以适应 solve_ivp
    return np.concatenate ([dUdt, dVdt])


# Initial conditions
Z0 = np.concatenate ([U0.reshape (n2), V0.reshape (n2)])

# Solve the system
sol = solve_ivp (rhs, (t_span[0], t_span[-1]), Z0, t_eval=t_span, method='RK45')

A2 = sol.y

print (A2.shape)
print (A2)
