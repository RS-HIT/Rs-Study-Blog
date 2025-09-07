"""
神经网络核心概念详解
这个文件详细解释了神经网络的基本概念和数学原理
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_neuron():
    """解释神经元的工作原理"""
    print("=" * 50)
    print("1. 神经元 (Neuron)")
    print("=" * 50)
    
    print("""
神经元是神经网络的基本计算单元，模仿生物神经元的工作方式。

神经元的计算过程：
1. 接收多个输入 (x1, x2, ..., xn)
2. 每个输入都有对应的权重 (w1, w2, ..., wn)
3. 计算加权和：z = w1*x1 + w2*x2 + ... + wn*xn + b
4. 通过激活函数处理：output = f(z)
    """)
    
    # 示例：单个神经元的计算
    print("示例：单个神经元计算")
    print("-" * 30)
    
    # 输入数据
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, -0.3, 0.8])
    bias = 0.1
    
    # 计算加权和
    weighted_sum = np.dot(inputs, weights) + bias
    print(f"输入: {inputs}")
    print(f"权重: {weights}")
    print(f"偏置: {bias}")
    print(f"加权和: {weighted_sum:.4f}")
    
    # 应用Sigmoid激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    output = sigmoid(weighted_sum)
    print(f"经过Sigmoid激活函数后的输出: {output:.4f}")
    

def explain_layers():
    """解释神经网络的层结构"""
    print("\n" + "=" * 50)
    print("2. 神经网络的层结构")
    print("=" * 50)
    
    print("""
神经网络由多层神经元组成：

1. 输入层 (Input Layer):
   - 接收原始数据
   - 神经元数量等于特征数量
   - 不进行计算，只是传递数据

2. 隐藏层 (Hidden Layer):
   - 处理和转换数据
   - 可以有多个隐藏层
   - 每层的神经元数量可以不同
   - 这里发生主要的学习过程

3. 输出层 (Output Layer):
   - 产生最终预测结果
   - 神经元数量取决于任务类型
   - 分类任务：神经元数量 = 类别数量
   - 回归任务：通常只有1个神经元
    """)
    
    # 可视化网络结构
    print("示例：3层神经网络结构 (输入层-隐藏层-输出层)")
    print("输入层(4个神经元) -> 隐藏层(6个神经元) -> 输出层(3个神经元)")


def explain_activation_functions():
    """解释激活函数"""
    print("\n" + "=" * 50)
    print("3. 激活函数 (Activation Functions)")
    print("=" * 50)
    
    print("""
激活函数的作用：
1. 引入非线性：让神经网络能够学习复杂的模式
2. 控制输出范围：将输出限制在特定范围内
3. 决定神经元是否被"激活"

常用激活函数：
    """)
    
    # 生成数据点用于绘制
    x = np.linspace(-5, 5, 100)
    
    # 定义激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def softmax_demo():
        # Softmax示例
        x = np.array([2.0, 1.0, 0.1])
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        print(f"Softmax示例:")
        print(f"输入: {x}")
        print(f"Softmax输出: {softmax_x}")
        print(f"输出和: {np.sum(softmax_x):.6f}")
        return softmax_x
    
    # 绘制激活函数
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('常用激活函数', fontsize=16)
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid函数')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True)
    axes[0, 0].text(-4, 0.8, '优点：输出范围[0,1]\n缺点：梯度消失问题', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Tanh
    axes[0, 1].plot(x, tanh(x), 'r-', linewidth=2)
    axes[0, 1].set_title('Tanh函数')
    axes[0, 1].grid(True)
    axes[0, 1].text(-4, 0.5, '优点：输出范围[-1,1]\n缺点：梯度消失问题', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # ReLU
    axes[1, 0].plot(x, relu(x), 'g-', linewidth=2)
    axes[1, 0].set_title('ReLU函数')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].grid(True)
    axes[1, 0].text(-4, 3, '优点：计算简单，无梯度消失\n缺点：神经元可能"死亡"', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Softmax示例
    categories = ['猫', '狗', '鸟']
    softmax_values = softmax_demo()
    axes[1, 1].bar(categories, softmax_values, color=['orange', 'skyblue', 'lightgreen'])
    axes[1, 1].set_title('Softmax函数(多分类)')
    axes[1, 1].set_ylabel('概率')
    for i, v in enumerate(softmax_values):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()


def explain_forward_propagation():
    """解释前向传播"""
    print("\n" + "=" * 50)
    print("4. 前向传播 (Forward Propagation)")
    print("=" * 50)
    
    print("""
前向传播是数据在神经网络中从输入层到输出层的流动过程：

步骤：
1. 输入数据进入输入层
2. 每一层对数据进行线性变换: z = W*x + b
3. 应用激活函数: a = f(z)
4. 将激活值传递到下一层
5. 重复步骤2-4，直到输出层
6. 得到最终预测结果

数学表示：
- Layer 1: z¹ = W¹x + b¹, a¹ = f(z¹)
- Layer 2: z² = W²a¹ + b², a² = f(z²)
- ...
    """)
    
    # 前向传播示例
    print("\n前向传播示例：")
    print("-" * 20)
    
    # 简单的2层网络
    # 输入
    X = np.array([[1.0, 2.0]])  # 1个样本，2个特征
    print(f"输入 X: {X}")
    
    # 第一层权重和偏置
    W1 = np.array([[0.5, 0.3, 0.2],
                   [0.1, 0.8, 0.6]])  # 2x3 矩阵
    b1 = np.array([[0.1, 0.2, 0.3]])  # 1x3 矩阵
    
    # 第一层前向传播
    z1 = np.dot(X, W1) + b1
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid激活
    
    print(f"第一层线性输出 z1: {z1}")
    print(f"第一层激活输出 a1: {a1}")
    
    # 第二层权重和偏置
    W2 = np.array([[0.9],
                   [0.4],
                   [0.7]])  # 3x1 矩阵
    b2 = np.array([[0.5]])   # 1x1 矩阵
    
    # 第二层前向传播
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid激活
    
    print(f"第二层线性输出 z2: {z2}")
    print(f"最终输出 a2: {a2}")


def explain_loss_function():
    """解释损失函数"""
    print("\n" + "=" * 50)
    print("5. 损失函数 (Loss Function)")
    print("=" * 50)
    
    print("""
损失函数衡量模型预测值与真实值之间的差距：

常用损失函数：
1. 均方误差 (MSE) - 用于回归任务
   Loss = (1/2) * (y_true - y_pred)²

2. 交叉熵损失 - 用于分类任务
   Loss = -Σ y_true * log(y_pred)

3. 二元交叉熵 - 用于二分类
   Loss = -[y*log(p) + (1-y)*log(1-p)]
    """)
    
    # 损失函数示例
    print("\n损失函数示例：")
    print("-" * 20)
    
    # 回归任务 - MSE
    y_true_reg = 5.0
    y_pred_reg = 4.2
    mse_loss = 0.5 * (y_true_reg - y_pred_reg) ** 2
    print(f"回归任务 - 真实值: {y_true_reg}, 预测值: {y_pred_reg}")
    print(f"MSE损失: {mse_loss:.4f}")
    
    # 分类任务 - 交叉熵
    y_true_cls = np.array([0, 1, 0])  # 真实标签：类别1
    y_pred_cls = np.array([0.1, 0.8, 0.1])  # 预测概率
    cross_entropy_loss = -np.sum(y_true_cls * np.log(y_pred_cls + 1e-12))
    print(f"\n分类任务 - 真实标签: {y_true_cls}, 预测概率: {y_pred_cls}")
    print(f"交叉熵损失: {cross_entropy_loss:.4f}")


def explain_backpropagation():
    """解释反向传播"""
    print("\n" + "=" * 50)
    print("6. 反向传播 (Backpropagation)")
    print("=" * 50)
    
    print("""
反向传播是训练神经网络的核心算法：

目的：根据损失函数计算每个参数的梯度，然后更新参数

步骤：
1. 计算输出层的误差
2. 将误差反向传播到前面的层
3. 计算每层参数的梯度
4. 使用梯度更新参数

数学原理：
- 使用链式法则计算偏导数
- ∂Loss/∂W = ∂Loss/∂a * ∂a/∂z * ∂z/∂W

参数更新：
- W_new = W_old - learning_rate * ∂Loss/∂W
- b_new = b_old - learning_rate * ∂Loss/∂b
    """)
    
    print("\n简单的反向传播示例：")
    print("-" * 30)
    print("考虑单个神经元：y = sigmoid(w*x + b)")
    
    # 简单示例
    x, y_true = 2.0, 1.0  # 输入和真实输出
    w, b = 0.5, 0.1       # 初始权重和偏置
    learning_rate = 0.1
    
    # 前向传播
    z = w * x + b
    y_pred = 1 / (1 + np.exp(-z))  # Sigmoid
    loss = 0.5 * (y_true - y_pred) ** 2  # MSE损失
    
    print(f"前向传播：")
    print(f"  z = {w} * {x} + {b} = {z}")
    print(f"  y_pred = sigmoid({z}) = {y_pred:.4f}")
    print(f"  loss = 0.5 * ({y_true} - {y_pred:.4f})² = {loss:.4f}")
    
    # 反向传播计算梯度
    # 链式法则：∂Loss/∂w = ∂Loss/∂y_pred * ∂y_pred/∂z * ∂z/∂w
    dLoss_dy_pred = -(y_true - y_pred)
    dy_pred_dz = y_pred * (1 - y_pred)  # Sigmoid导数
    dz_dw = x
    dz_db = 1
    
    dLoss_dw = dLoss_dy_pred * dy_pred_dz * dz_dw
    dLoss_db = dLoss_dy_pred * dy_pred_dz * dz_db
    
    print(f"\n反向传播：")
    print(f"  ∂Loss/∂w = {dLoss_dw:.4f}")
    print(f"  ∂Loss/∂b = {dLoss_db:.4f}")
    
    # 更新参数
    w_new = w - learning_rate * dLoss_dw
    b_new = b - learning_rate * dLoss_db
    
    print(f"\n参数更新：")
    print(f"  w: {w} -> {w_new:.4f}")
    print(f"  b: {b} -> {b_new:.4f}")


def explain_training_process():
    """解释训练过程"""
    print("\n" + "=" * 50)
    print("7. 神经网络训练过程")
    print("=" * 50)
    
    print("""
神经网络的训练是一个迭代过程：

1. 初始化权重和偏置（通常使用随机值）
2. 对于每个训练周期（Epoch）：
   a. 前向传播：计算预测值
   b. 计算损失：比较预测值和真实值
   c. 反向传播：计算梯度
   d. 更新参数：使用梯度下降
3. 重复步骤2，直到达到预定条件

关键概念：
- Epoch：完整遍历一次训练数据集
- Batch：一次处理的样本数量
- Learning Rate：学习率，控制参数更新的步长
- Gradient Descent：梯度下降优化算法

训练监控：
- 训练损失：监控模型在训练数据上的表现
- 验证损失：监控模型的泛化能力
- 准确率：分类任务的评估指标
    """)
    
    # 简单的训练过程可视化
    print("\n模拟训练过程：")
    print("-" * 20)
    
    epochs = np.arange(1, 11)
    train_loss = np.array([2.5, 1.8, 1.3, 1.0, 0.8, 0.7, 0.6, 0.55, 0.52, 0.50])
    train_acc = np.array([0.3, 0.45, 0.6, 0.7, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('训练损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(epochs, train_acc, 'r-o', linewidth=2, markersize=6)
    ax2.set_title('训练准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    for i, epoch in enumerate(epochs):
        print(f"Epoch {epoch}: Loss = {train_loss[i]:.2f}, Accuracy = {train_acc[i]:.2f}")


def main():
    """主函数：运行所有概念解释"""
    print("神经网络核心概念详解")
    print("这个程序将详细解释神经网络的基本概念和工作原理\n")
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 依次解释各个概念
    explain_neuron()
    explain_layers()
    explain_activation_functions()
    explain_forward_propagation()
    explain_loss_function()
    explain_backpropagation()
    explain_training_process()
    
    print("\n" + "=" * 50)
    print("总结")
    print("=" * 50)
    print("""
神经网络的核心思想：
1. 模仿大脑神经元的工作方式
2. 通过大量数据学习输入和输出之间的复杂关系
3. 使用梯度下降算法不断优化参数
4. 具有强大的函数拟合和模式识别能力

神经网络的强大之处：
- 通用近似定理：理论上可以拟合任意复杂的函数
- 自动特征学习：不需要手动设计特征
- 端到端学习：从原始数据直接学习到最终输出
- 可扩展性：可以通过增加层数和神经元数量提高表达能力

现在你可以运行 simple_neural_network.py 来实践这些概念！
    """)


if __name__ == "__main__":
    main()
