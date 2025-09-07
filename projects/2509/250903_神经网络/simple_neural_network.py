"""
简单神经网络实现 - 手写数字识别
这个例子帮助理解神经网络的基本概念：神经元、层、前向传播、反向传播
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

class SimpleNeuralNetwork:
    """
    简单的三层神经网络实现
    输入层 -> 隐藏层 -> 输出层
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化神经网络
        :param input_size: 输入层神经元数量
        :param hidden_size: 隐藏层神经元数量  
        :param output_size: 输出层神经元数量
        :param learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 随机初始化权重和偏置
        # 输入层到隐藏层的权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        # 隐藏层到输出层的权重
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # 记录训练过程
        self.train_losses = []
        self.train_accuracies = []
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        # 防止数值溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax激活函数，用于多分类输出"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        前向传播：数据从输入层流向输出层
        """
        # 输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = self.sigmoid(self.z1)          # 激活函数
        
        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 线性变换
        self.a2 = self.softmax(self.z2)               # Softmax激活
        
        return self.a2
    
    def backward_propagation(self, X, y, output):
        """
        反向传播：计算梯度并更新权重
        """
        m = X.shape[0]  # 样本数量
        
        # 输出层误差
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # 隐藏层误差
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # 更新权重和偏置
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        """计算交叉熵损失"""
        m = y_true.shape[0]
        # 避免log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """计算准确率"""
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    
    def train(self, X, y, epochs=1000, print_every=100):
        """
        训练神经网络
        """
        print(f"开始训练神经网络...")
        print(f"网络结构: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        print(f"训练样本数: {X.shape[0]}")
        print("="*50)
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward_propagation(X)
            
            # 计算损失和准确率
            loss = self.compute_loss(y, output)
            accuracy = self.compute_accuracy(y, output)
            
            # 记录训练过程
            self.train_losses.append(loss)
            self.train_accuracies.append(accuracy)
            
            # 反向传播
            self.backward_propagation(X, y, output)
            
            # 打印训练进度
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        print("训练完成！")
    
    def predict(self, X):
        """预测"""
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        return self.forward_propagation(X)


def prepare_data():
    """准备手写数字数据集"""
    print("正在加载手写数字数据集...")
    
    # 加载sklearn内置的手写数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"数据集信息:")
    print(f"- 样本数量: {X.shape[0]}")
    print(f"- 特征维度: {X.shape[1]} (8x8像素图片)")
    print(f"- 类别数量: {len(np.unique(y))} (数字0-9)")
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 将标签转换为one-hot编码
    def to_one_hot(y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    y_one_hot = to_one_hot(y, 10)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, digits


def visualize_samples(digits, num_samples=10):
    """可视化一些手写数字样本"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('手写数字样本', fontsize=16)
    
    for i in range(num_samples):
        row = i // 5
        col = i % 5
        
        # 显示图片
        axes[row, col].imshow(digits.images[i], cmap='gray')
        axes[row, col].set_title(f'数字: {digits.target[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(nn):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失函数图
    ax1.plot(nn.train_losses)
    ax1.set_title('训练损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率图
    ax2.plot(nn.train_accuracies)
    ax2.set_title('训练准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_predictions(nn, X_test, y_test, digits, num_samples=10):
    """测试预测结果并可视化"""
    # 预测
    predictions = nn.predict(X_test)
    probabilities = nn.predict_proba(X_test)
    true_labels = np.argmax(y_test, axis=1)
    
    # 计算测试准确率
    test_accuracy = np.mean(predictions == true_labels)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 可视化一些预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    fig.suptitle(f'预测结果示例 (测试准确率: {test_accuracy:.4f})', fontsize=16)
    
    for i in range(num_samples):
        row = i // 5
        col = i % 5
        
        # 获取原始图片数据进行显示
        sample_idx = i
        original_image = X_test[sample_idx].reshape(8, 8)
        
        # 显示图片
        axes[row, col].imshow(original_image, cmap='gray')
        
        pred_label = predictions[sample_idx]
        true_label = true_labels[sample_idx]
        confidence = np.max(probabilities[sample_idx])
        
        # 设置标题，正确预测用绿色，错误预测用红色
        color = 'green' if pred_label == true_label else 'red'
        title = f'真实: {true_label}, 预测: {pred_label}\n置信度: {confidence:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("           简单神经网络 - 手写数字识别")
    print("="*60)
    
    # 1. 准备数据
    X_train, X_test, y_train, y_test, digits = prepare_data()
    
    # 2. 可视化一些样本
    print("\n正在显示手写数字样本...")
    visualize_samples(digits)
    
    # 3. 创建神经网络
    print("\n创建神经网络...")
    input_size = X_train.shape[1]  # 64个特征 (8x8像素)
    hidden_size = 32               # 隐藏层32个神经元
    output_size = 10               # 10个输出 (数字0-9)
    
    nn = SimpleNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size, 
        output_size=output_size,
        learning_rate=0.1
    )
    
    # 4. 训练神经网络
    print("\n开始训练...")
    start_time = time.time()
    nn.train(X_train, y_train, epochs=500, print_every=100)
    training_time = time.time() - start_time
    print(f"训练用时: {training_time:.2f}秒")
    
    # 5. 可视化训练过程
    print("\n显示训练历史...")
    plot_training_history(nn)
    
    # 6. 测试神经网络
    print("\n测试神经网络...")
    test_predictions(nn, X_test, y_test, digits)
    
    print("\n" + "="*60)
    print("神经网络基本概念总结:")
    print("1. 神经元: 接收输入，计算加权和，通过激活函数输出")
    print("2. 层: 多个神经元组成一层，信息逐层传递")
    print("3. 权重: 连接神经元的参数，通过学习调整")
    print("4. 前向传播: 输入数据从输入层流向输出层")
    print("5. 反向传播: 根据误差反向调整权重")
    print("6. 激活函数: 增加非线性，让网络能学习复杂模式")
    print("="*60)


if __name__ == "__main__":
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
