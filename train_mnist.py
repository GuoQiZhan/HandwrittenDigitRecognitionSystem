#!/usr/bin/env python3
"""
MNIST手写数字识别模型训练脚本。
训练一个简单的CNN模型，并将其保存为.pth格式。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
import time
from datetime import datetime

# 导入项目中的模型定义
from src.model_loader import MNISTCNN


def train_model(epochs=10, batch_size=64, learning_rate=0.001, save_path="models/mnist_model.pth"):
    """训练MNIST模型。

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 模型保存路径
    """
    # 确保模型目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    print("加载MNIST数据集...")
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建模型
    model = MNISTCNN().to(device)
    print(f"模型架构:\n{model}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    print(f"\n开始训练，共{epochs}个epoch...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_accuracy = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_accuracy = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        # 打印epoch结果
        print(f'\nEpoch {epoch} 结果:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%')

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f}秒")

    # 保存模型
    print(f"保存模型到: {save_path}")
    torch.save(model.state_dict(), save_path)

    # 也保存完整模型（可选）
    full_model_path = save_path.replace('.pth', '_full.pth')
    torch.save(model, full_model_path)
    print(f"保存完整模型到: {full_model_path}")

    return model


def evaluate_model(model_path="models/mnist_model.pth"):
    """评估训练好的模型。

    Args:
        model_path: 模型文件路径
    """
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"\n评估模型: {model_path}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = MNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 评估
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f"模型在测试集上的准确率: {accuracy:.2f}%")

    # 逐类别准确率
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            c = predicted.eq(target).squeeze()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n每个数字的准确率:")
    for i in range(10):
        if class_total[i] > 0:
            print(f'  数字 {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'  数字 {i}: 无样本')


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='MNIST手写数字识别模型训练')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--model-path', type=str, default='models/mnist_model.pth',
                        help='模型保存/加载路径')

    args = parser.parse_args()

    # 检查参数
    if not (args.train or args.evaluate):
        parser.print_help()
        print("\n请指定 --train 或 --evaluate 参数")
        return

    # 执行训练或评估
    if args.train:
        print("=" * 60)
        print("MNIST模型训练")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.model_path
        )

    if args.evaluate:
        evaluate_model(args.model_path)


if __name__ == "__main__":
    main()