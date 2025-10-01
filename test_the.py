import matplotlib.pyplot as plt

eps = 0.3
T = 1000000
B = T**(2*eps)
eta = 0.9
p = 1/10**5
mu_l = 1000
mu_r = 5
delta_1 = 15
L = 1500

import numpy as np
for alpha in np.arange(0.05, 1.05, 0.05):
    print(f"alpha: {alpha}")
    for beta in np.arange(1, 2.05, 0.05):

        # 假设 bar_alpha_i, bar_beta_i, mu_i_l, mu_i_r, eta_i, p_i, L, delta_1, h_i, g_i 已经定义
        # 这里只是示例，实际使用时请根据实际变量替换
        bar_alpha = alpha
        bar_beta = beta

        def h(x):
            return 240*x/(eta*p) + 5

        def g(x):
            return x**1.2/(eta*p) + 1
        # 计算括号内的正部
        log_term = delta_1 + np.log(h(bar_alpha * eta * p) * mu_r) \
            - np.log(bar_beta * g(bar_alpha * eta * p) * mu_l) \
            - (bar_beta * mu_l) / L
        print(f"Log Term: {log_term:.2f}", f"Log h: {np.log(h(bar_alpha * eta * p) * mu_r):.2f}", f"Log g: {np.log(bar_beta * g(bar_alpha * eta * p) * mu_l):.2f}", f"Log (beta * mu_l / L): {np.log((bar_beta * mu_l) / L):.2f}")
        
        positive_part = max(log_term, 0)

        U_1 = B * eta * p * (bar_beta * mu_l - bar_alpha * mu_l) \
            + B * eta * p * L \
            + B * eta * p * L * positive_part

        
        if 'U1_values' not in locals():
            U1_values = []
            alpha_values = []
            beta_values = []

        U1_values.append(U_1)
        alpha_values.append(alpha)
        beta_values.append(beta)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 转换为numpy数组以便reshape
alpha_arr = np.array(alpha_values)
beta_arr = np.array(beta_values)
U1_arr = np.array(U1_values)

# 由于alpha和beta是网格遍历的，可以reshape为meshgrid形状
alpha_unique = np.unique(alpha_arr)
beta_unique = np.unique(beta_arr)
alpha_grid, beta_grid = np.meshgrid(alpha_unique, beta_unique, indexing='ij')
U1_grid = U1_arr.reshape(len(alpha_unique), len(beta_unique))

surf = ax.plot_surface(alpha_grid, beta_grid, U1_grid, cmap='viridis')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='U_1')
ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('U_1')
ax.set_title('U_1 vs alpha beta')
plt.tight_layout()
plt.show()

# 找到最优alpha和beta
optimal_idx = np.argmax(U1_arr)
optimal_alpha = alpha_arr[optimal_idx]
optimal_beta = beta_arr[optimal_idx]
optimal_U1 = U1_arr[optimal_idx]

print(f"最优解: alpha = {optimal_alpha:.4f}, beta = {optimal_beta:.4f}, U_1 = {optimal_U1:.4f}")

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1: 固定最优alpha，枚举beta
optimal_alpha_mask = np.abs(alpha_arr - optimal_alpha) < 1e-10
beta_values_at_optimal_alpha = beta_arr[optimal_alpha_mask]
U1_values_at_optimal_alpha = U1_arr[optimal_alpha_mask]

# Sort by beta
sorted_indices = np.argsort(beta_values_at_optimal_alpha)
beta_sorted = beta_values_at_optimal_alpha[sorted_indices]
U1_sorted = U1_values_at_optimal_alpha[sorted_indices]

ax1.plot(beta_sorted, U1_sorted, 'b-', linewidth=2, label=f'alpha = {optimal_alpha:.4f}')
ax1.axvline(x=optimal_beta, color='r', linestyle='--', alpha=0.7, label=f'Optimal beta = {optimal_beta:.4f}')
ax1.set_xlabel('beta')
ax1.set_ylabel('U_1')
ax1.set_title('U_1 vs beta with fixed optimal alpha')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Fixed optimal beta, enumerate alpha
optimal_beta_mask = np.abs(beta_arr - optimal_beta) < 1e-10
alpha_values_at_optimal_beta = alpha_arr[optimal_beta_mask]
U1_values_at_optimal_beta = U1_arr[optimal_beta_mask]

# Sort by alpha
sorted_indices = np.argsort(alpha_values_at_optimal_beta)
alpha_sorted = alpha_values_at_optimal_beta[sorted_indices]
U1_sorted = U1_values_at_optimal_beta[sorted_indices]

ax2.plot(alpha_sorted, U1_sorted, 'g-', linewidth=2, label=f'beta = {optimal_beta:.4f}')
ax2.axvline(x=optimal_alpha, color='r', linestyle='--', alpha=0.7, label=f'Optimal alpha = {optimal_alpha:.4f}')
ax2.set_xlabel('alpha')
ax2.set_ylabel('U_1')
ax2.set_title('U_1 vs alpha with fixed optimal beta')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
