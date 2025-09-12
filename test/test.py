import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义范围
# 1 < x < 3
# -2 < y < -1
# 求 xy + x - y 的范围

# 创建网格点
x = np.linspace(1.001, 2.999, 100)  # 稍微缩小范围避免边界
y = np.linspace(-1.999, -1.001, 100)
X, Y = np.meshgrid(x, y)

# 计算函数值 f(x,y) = xy + x - y
Z = X * Y + X - Y

# 1. 理论分析：找到边界值
print("理论分析：")
print("约束条件：1 < x < 3, -2 < y < -1")
print("目标函数：f(x,y) = xy + x - y")
print()

# 分析函数性质
print("函数对x的偏导数：∂f/∂x = y + 1")
print("函数对y的偏导数：∂f/∂y = x - 1")
print()

# 在给定范围内，分析偏导数的符号
print("在给定范围内：")
print("- 当 -2 < y < -1 时，y + 1 ∈ (-1, 0)，所以 ∂f/∂x < 0")
print("- 当 1 < x < 3 时，x - 1 ∈ (0, 2)，所以 ∂f/∂y > 0")
print()
print("这意味着：")
print("- f(x,y) 关于 x 单调递减")
print("- f(x,y) 关于 y 单调递增")
print()

# 计算边界值
corner_values = []
corners = [(1, -2), (1, -1), (3, -2), (3, -1)]

print("计算四个角点的函数值（趋近值）：")
for x_val, y_val in corners:
    f_val = x_val * y_val + x_val - y_val
    corner_values.append(f_val)
    print(f"f({x_val}, {y_val}) = {x_val}×{y_val} + {x_val} - ({y_val}) = {f_val}")

print()
print(f"最小值趋近于：{min(corner_values)} （在点 (3, -2) 附近）")
print(f"最大值趋近于：{max(corner_values)} （在点 (1, -1) 附近）")
print(f"因此，xy + x - y 的取值范围是：({min(corner_values)}, {max(corner_values)})")

# 计算实际数值范围
z_min = np.min(Z)
z_max = np.max(Z)
print(f"\n数值计算结果：")
print(f"最小值：{z_min:.6f}")
print(f"最大值：{z_max:.6f}")
print(f"范围：({z_min:.6f}, {z_max:.6f})")

# 2. 可视化
fig = plt.figure(figsize=(15, 10))

# 子图1：3D表面图
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y) = xy + x - y')
ax1.set_title('3D Surface Plot of f(x,y) = xy + x - y')
plt.colorbar(surf, ax=ax1, shrink=0.5)

# 子图2：等高线图
ax2 = fig.add_subplot(222)
contour = ax2.contour(X, Y, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot of f(x,y) = xy + x - y')
ax2.grid(True)

# 子图3：填充等高线图
ax3 = fig.add_subplot(223)
contourf = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contourf, ax=ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Filled Contour Plot')

# 子图4：函数值分布直方图
ax4 = fig.add_subplot(224)
ax4.hist(Z.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax4.set_xlabel('f(x,y) = xy + x - y')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Function Values')
ax4.axvline(z_min, color='red', linestyle='--', label=f'Min: {z_min:.3f}')
ax4.axvline(z_max, color='red', linestyle='--', label=f'Max: {z_max:.3f}')
ax4.legend()

plt.tight_layout()
plt.savefig('function_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 更详细的边界分析
print("\n详细边界分析：")
print("由于函数在区域内的单调性，极值出现在区域边界上")

# 分析四条边界
print("\n四条边界上的函数值：")

# 边界1：x → 1
x_edge1 = 1
y_edge1 = np.linspace(-1.999, -1.001, 100)
z_edge1 = x_edge1 * y_edge1 + x_edge1 - y_edge1
print(f"边界 x→1: f值范围 [{np.min(z_edge1):.6f}, {np.max(z_edge1):.6f}]")

# 边界2：x → 3
x_edge2 = 3
y_edge2 = np.linspace(-1.999, -1.001, 100)
z_edge2 = x_edge2 * y_edge2 + x_edge2 - y_edge2
print(f"边界 x→3: f值范围 [{np.min(z_edge2):.6f}, {np.max(z_edge2):.6f}]")

# 边界3：y → -2
x_edge3 = np.linspace(1.001, 2.999, 100)
y_edge3 = -2
z_edge3 = x_edge3 * y_edge3 + x_edge3 - y_edge3
print(f"边界 y→-2: f值范围 [{np.min(z_edge3):.6f}, {np.max(z_edge3):.6f}]")

# 边界4：y → -1
x_edge4 = np.linspace(1.001, 2.999, 100)
y_edge4 = -1
z_edge4 = x_edge4 * y_edge4 + x_edge4 - y_edge4
print(f"边界 y→-1: f值范围 [{np.min(z_edge4):.6f}, {np.max(z_edge4):.6f}]")

# 创建边界图
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 四条边界的函数值
ax1.plot(y_edge1, z_edge1, 'b-', linewidth=2)
ax1.set_xlabel('y')
ax1.set_ylabel('f(1, y)')
ax1.set_title('Boundary: x → 1')
ax1.grid(True)

ax2.plot(y_edge2, z_edge2, 'r-', linewidth=2)
ax2.set_xlabel('y')
ax2.set_ylabel('f(3, y)')
ax2.set_title('Boundary: x → 3')
ax2.grid(True)

ax3.plot(x_edge3, z_edge3, 'g-', linewidth=2)
ax3.set_xlabel('x')
ax3.set_ylabel('f(x, -2)')
ax3.set_title('Boundary: y → -2')
ax3.grid(True)

ax4.plot(x_edge4, z_edge4, 'm-', linewidth=2)
ax4.set_xlabel('x')
ax4.set_ylabel('f(x, -1)')
ax4.set_title('Boundary: y → -1')
ax4.grid(True)

plt.tight_layout()
plt.savefig('boundary_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n结论：")
print(f"在约束条件 1 < x < 3, -2 < y < -1 下")
print(f"函数 f(x,y) = xy + x - y 的取值范围是 ({min(corner_values)}, {max(corner_values)})")
