"""
交互式神经网络演示
让用户可以调整参数，实时看到神经网络行为的变化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

class InteractiveNeuralNetwork:
    """交互式神经网络演示类"""
    
    def __init__(self):
        self.data_type = "classification"
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()
        
    def generate_data(self, data_type="classification", n_samples=200, noise=0.1):
        """生成演示数据"""
        self.data_type = data_type
        
        if data_type == "classification":
            # 生成分类数据
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_clusters_per_class=1,
                random_state=42
            )
            # 转换为二分类
            y = (y == 1).astype(int)
            
        else:
            # 生成回归数据
            X, y = make_regression(
                n_samples=n_samples,
                n_features=2,
                noise=noise * 10,
                random_state=42
            )
        
        # 标准化数据
        X = self.scaler.fit_transform(X)
        self.X_train = X
        self.y_train = y
        
        return X, y
    
    def visualize_data(self, X, y, title="数据分布"):
        """可视化数据"""
        plt.figure(figsize=(8, 6))
        
        if self.data_type == "classification":
            colors = ['red' if label == 0 else 'blue' for label in y]
            labels = ['类别 0' if label == 0 else '类别 1' for label in y]
            scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
            # 添加图例
            red_patch = patches.Patch(color='red', label='类别 0')
            blue_patch = patches.Patch(color='blue', label='类别 1')
            plt.legend(handles=[red_patch, blue_patch])
        else:
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='目标值')
            
        plt.title(title)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def single_neuron_demo(self):
        """单个神经元演示"""
        print("=" * 50)
        print("单个神经元演示")
        print("=" * 50)
        
        # 生成简单的2D分类数据
        X, y = self.generate_data("classification", n_samples=100)
        self.visualize_data(X, y, "原始数据分布")
        
        print("\n单个神经元进行二分类...")
        print("神经元输出 = sigmoid(w1*x1 + w2*x2 + b)")
        
        # 手动训练单个神经元
        def train_single_neuron(X, y, epochs=100, lr=0.1):
            # 初始化参数
            w = np.random.randn(2) * 0.5
            b = 0.0
            
            losses = []
            
            for epoch in range(epochs):
                # 前向传播
                z = np.dot(X, w) + b
                predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
                
                # 计算损失
                loss = -np.mean(y * np.log(predictions + 1e-12) + 
                               (1 - y) * np.log(1 - predictions + 1e-12))
                losses.append(loss)
                
                # 反向传播
                dw = np.dot(X.T, (predictions - y)) / len(y)
                db = np.mean(predictions - y)
                
                # 更新参数
                w -= lr * dw
                b -= lr * db
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            return w, b, losses
        
        # 训练神经元
        w, b, losses = train_single_neuron(X, y)
        
        print(f"\n训练完成！")
        print(f"最终权重: w1={w[0]:.3f}, w2={w[1]:.3f}")
        print(f"最终偏置: b={b:.3f}")
        
        # 可视化决策边界
        self.plot_decision_boundary(X, y, w, b, "单个神经元的决策边界")
        
        # 绘制损失曲线
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.title('单个神经元训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    def plot_decision_boundary(self, X, y, w, b, title):
        """绘制决策边界"""
        plt.figure(figsize=(10, 6))
        
        # 绘制数据点
        colors = ['red' if label == 0 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # 计算决策边界
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        z = np.dot(grid_points, w) + b
        predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        predictions = predictions.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contour(xx, yy, predictions, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        plt.contourf(xx, yy, predictions, levels=50, alpha=0.3, cmap='RdBu')
        
        plt.title(title)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        
        # 添加图例
        red_patch = patches.Patch(color='red', label='类别 0')
        blue_patch = patches.Patch(color='blue', label='类别 1')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()
    
    def activation_function_comparison(self):
        """激活函数比较演示"""
        print("\n" + "=" * 50)
        print("激活函数比较演示")
        print("=" * 50)
        
        x = np.linspace(-5, 5, 1000)
        
        # 定义激活函数
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def tanh(x):
            return np.tanh(x)
        
        def relu(x):
            return np.maximum(0, x)
        
        def leaky_relu(x, alpha=0.01):
            return np.where(x > 0, x, alpha * x)
        
        # 绘制激活函数及其导数
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('激活函数及其导数比较', fontsize=16)
        
        functions = [
            (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)), 'Sigmoid'),
            (tanh, lambda x: 1 - tanh(x)**2, 'Tanh'),
            (relu, lambda x: (x > 0).astype(float), 'ReLU'),
            (leaky_relu, lambda x: np.where(x > 0, 1, 0.01), 'Leaky ReLU')
        ]
        
        for i, (func, derivative, name) in enumerate(functions):
            # 绘制函数
            axes[0, i].plot(x, func(x), 'b-', linewidth=2, label=name)
            axes[0, i].set_title(f'{name} 函数')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # 绘制导数
            axes[1, i].plot(x, derivative(x), 'r-', linewidth=2, label=f'{name} 导数')
            axes[1, i].set_title(f'{name} 导数')
            axes[1, i].set_xlabel('x')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\n激活函数特点：")
        print("1. Sigmoid: 输出范围[0,1]，存在梯度消失问题")
        print("2. Tanh: 输出范围[-1,1]，比Sigmoid好一些")
        print("3. ReLU: 计算简单，解决梯度消失，但可能出现神经元死亡")
        print("4. Leaky ReLU: 解决ReLU的神经元死亡问题")
    
    def learning_rate_demo(self):
        """学习率对训练的影响演示"""
        print("\n" + "=" * 50)
        print("学习率对训练影响演示")
        print("=" * 50)
        
        # 生成简单的回归数据
        X, y = self.generate_data("regression", n_samples=100, noise=0.1)
        
        # 不同学习率
        learning_rates = [0.001, 0.01, 0.1, 1.0]
        
        plt.figure(figsize=(15, 10))
        
        for i, lr in enumerate(learning_rates):
            print(f"\n测试学习率: {lr}")
            
            # 训练简单的线性模型
            w = np.random.randn(2) * 0.1
            b = 0.0
            losses = []
            
            for epoch in range(200):
                # 前向传播
                predictions = np.dot(X, w) + b
                loss = np.mean((predictions - y) ** 2)
                losses.append(loss)
                
                # 反向传播
                dw = 2 * np.dot(X.T, (predictions - y)) / len(y)
                db = 2 * np.mean(predictions - y)
                
                # 更新参数
                w -= lr * dw
                b -= lr * db
                
                # 检查是否发散
                if loss > 1e10:
                    print(f"  学习率 {lr} 导致训练发散！")
                    break
            
            # 绘制损失曲线
            plt.subplot(2, 2, i + 1)
            plt.plot(losses)
            plt.title(f'学习率 = {lr}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            final_loss = losses[-1] if losses else float('inf')
            print(f"  最终损失: {final_loss:.6f}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n学习率选择原则：")
        print("- 太小：训练很慢，可能陷入局部最优")
        print("- 太大：可能发散，无法收敛")
        print("- 合适：快速收敛到较好的解")
    
    def network_depth_demo(self):
        """网络深度对性能的影响演示"""
        print("\n" + "=" * 50)
        print("网络深度对性能影响演示")
        print("=" * 50)
        
        # 生成复杂的分类数据
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
        X = self.scaler.fit_transform(X)
        
        self.visualize_data(X, y, "复杂分类数据(同心圆)")
        
        print("这种同心圆数据需要非线性分类器...")
        print("比较不同深度的网络性能：")
        
        # 简化的网络训练函数
        def train_network(X, y, hidden_layers, epochs=200):
            """训练简单的前馈网络"""
            layers = [2] + hidden_layers + [1]  # 输入2维，输出1维
            
            # 初始化权重
            weights = []
            biases = []
            for i in range(len(layers) - 1):
                w = np.random.randn(layers[i], layers[i+1]) * 0.5
                b = np.zeros((1, layers[i+1]))
                weights.append(w)
                biases.append(b)
            
            losses = []
            
            for epoch in range(epochs):
                # 前向传播
                activations = [X]
                for i in range(len(weights)):
                    z = np.dot(activations[-1], weights[i]) + biases[i]
                    if i < len(weights) - 1:  # 隐藏层使用ReLU
                        a = np.maximum(0, z)
                    else:  # 输出层使用Sigmoid
                        a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                    activations.append(a)
                
                predictions = activations[-1].flatten()
                
                # 计算损失
                loss = -np.mean(y * np.log(predictions + 1e-12) + 
                               (1 - y) * np.log(1 - predictions + 1e-12))
                losses.append(loss)
                
                # 简化的反向传播（仅作演示）
                if epoch < epochs - 1:  # 最后一次不更新
                    lr = 0.01
                    # 输出层梯度
                    output_error = predictions - y
                    for i in range(len(weights) - 1, -1, -1):
                        if i == len(weights) - 1:
                            delta = output_error.reshape(-1, 1)
                        else:
                            # 简化的隐藏层梯度计算
                            delta = np.dot(delta, weights[i+1].T) * (activations[i+1] > 0)
                        
                        # 更新权重
                        weights[i] -= lr * np.dot(activations[i].T, delta) / len(y)
                        biases[i] -= lr * np.mean(delta, axis=0, keepdims=True)
            
            return losses, predictions
        
        # 测试不同网络结构
        network_configs = [
            ([], "线性分类器"),
            ([10], "1隐藏层(10神经元)"),
            ([20, 10], "2隐藏层(20,10神经元)"),
            ([30, 20, 10], "3隐藏层(30,20,10神经元)")
        ]
        
        plt.figure(figsize=(15, 10))
        
        for i, (hidden_layers, name) in enumerate(network_configs):
            print(f"\n训练 {name}...")
            losses, final_predictions = train_network(X, y, hidden_layers)
            
            # 计算准确率
            accuracy = np.mean((final_predictions > 0.5) == y)
            print(f"  最终准确率: {accuracy:.3f}")
            
            # 绘制损失曲线
            plt.subplot(2, 2, i + 1)
            plt.plot(losses)
            plt.title(f'{name}\n准确率: {accuracy:.3f}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\n网络深度的影响：")
        print("- 浅层网络：可能无法学习复杂模式")
        print("- 深层网络：更强的表达能力，但训练更困难")
        print("- 需要根据问题复杂度选择合适的网络深度")


def interactive_menu():
    """交互式菜单"""
    demo = InteractiveNeuralNetwork()
    
    while True:
        print("\n" + "=" * 60)
        print("           神经网络交互式演示菜单")
        print("=" * 60)
        print("1. 单个神经元演示")
        print("2. 激活函数比较")
        print("3. 学习率影响演示")
        print("4. 网络深度影响演示")
        print("5. 退出")
        print("=" * 60)
        
        choice = input("请选择演示内容 (1-5): ").strip()
        
        if choice == '1':
            demo.single_neuron_demo()
        elif choice == '2':
            demo.activation_function_comparison()
        elif choice == '3':
            demo.learning_rate_demo()
        elif choice == '4':
            demo.network_depth_demo()
        elif choice == '5':
            print("谢谢使用！希望这些演示帮助你更好地理解神经网络。")
            break
        else:
            print("无效选择，请重新输入。")
        
        input("\n按Enter键继续...")


def main():
    """主函数"""
    print("神经网络交互式演示程序")
    print("通过可视化帮助理解神经网络的各种概念")
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    interactive_menu()


if __name__ == "__main__":
    main()
