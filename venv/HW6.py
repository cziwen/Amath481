import numpy as np
from PIL.Image import Image
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义参数
L = 20  # 空间范围 [-L/2, L/2]
n = 64  # 网格点数
beta = 1
D1, D2 = 0.1, 0.1
t_span = np.arange(0, 4.5, 0.5)  # 时间范围
# t_eval = np.linspace(0, 4, 100)  # 输出的时间点

# 创建空间网格
x = np.linspace(-L / 2, L / 2, n, endpoint=False)
y = np.linspace(-L / 2, L / 2, n, endpoint=False)
dx = L / n
dy = L / n
X, Y = np.meshgrid(x, y)

# 初始条件
m = 1  # 螺旋数
angle = np.angle(X + 1j * Y)
radius = np.sqrt(X**2 + Y**2)
U0 = np.tanh(radius) * np.cos(m * angle - radius)
V0 = np.tanh(radius) * np.sin(m * angle - radius)

# 反应项
def lambda_A(U, V):
    A2 = U**2 + V**2
    return 1 - A2

def omega_A(U, V):
    A2 = U**2 + V**2
    return -beta * A2

# 使用FFT计算拉普拉斯算子
def laplacian_fft(U, dx, dy):
    U_hat = fft2(U)
    kx = 2 * np.pi * np.fft.fftfreq(U.shape[1], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(U.shape[0], d=dy)
    KX, KY = np.meshgrid(kx, ky)
    laplace_hat = -(KX**2 + KY**2) * U_hat
    return np.real(ifft2(laplace_hat))

# 定义方程右侧
def rhs(t, Z):
    U, V = Z[:n**2].reshape((n, n)), Z[n**2:].reshape((n, n))
    LU = laplacian_fft(U, dx, dy)
    LV = laplacian_fft(V, dx, dy)
    dUdt = lambda_A(U, V) * U - omega_A(U, V) * V + D1 * LU
    dVdt = omega_A(U, V) * U + lambda_A(U, V) * V + D2 * LV

    rhs = np.concatenate([dUdt.ravel(), dVdt.ravel()])
    return rhs

# 初始化
Z0 = np.concatenate([U0.ravel(), V0.ravel()])


# 数值求解
sol = solve_ivp(rhs,
                (t_span[0], t_span[-1]),
                Z0,
                t_eval=t_span,
                method='RK45')



# 可视化结果
sol_u = sol.y[:n**2]  # 提取 U 的解
sol_v = sol.y[n**2:]  # 提取 V 的解
n = int(np.sqrt(sol_u.shape[0]))  # 确定 n x n 的矩阵维度

# 创建绘图窗口
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
cax_u = axes[0].imshow(sol_u[:, 0].reshape((n, n)), extent=[-L/2, L/2, -L/2, L/2], cmap='jet')
cax_v = axes[1].imshow(sol_v[:, 0].reshape((n, n)), extent=[-L/2, L/2, -L/2, L/2], cmap='jet')

# 添加颜色条
fig.colorbar(cax_u, ax=axes[0], label='Field Intensity (U)')
fig.colorbar(cax_v, ax=axes[1], label='Field Intensity (V)')

# 设置子图标题和标签
axes[0].set_title('Field U at t = 0.00')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_title('Field V at t = 0.00')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

# 更新函数
def update(frame):
    t = t_span[frame]
    axes[0].set_title(f'Field U at t = {t:.2f}')
    axes[1].set_title(f'Field V at t = {t:.2f}')
    cax_u.set_data(sol_u[:, frame].reshape((n, n)))
    cax_v.set_data(sol_v[:, frame].reshape((n, n)))
    return cax_u, cax_v

# 创建动画
anim = FuncAnimation(fig, update, frames=len(t_span), blit=True)  # 使用 len(t_eval) 确定帧数

# 保存为 GIF
anim.save("/Users/ziwenchen/PycharmProjects/Amath481/reaction_diffusion_evolution_fft.gif", writer='pillow', fps=2)

print("GIF 生成完成，保存为 reaction_diffusion_evolution_fft.gif")