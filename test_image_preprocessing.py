#!/usr/bin/env python3
"""
测试画布图像预处理功能。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from src.image_processor import ImageProcessor
import torch


def test_canvas_preprocessing():
    """测试画布预处理。"""
    print("测试画布图像预处理...")

    # 创建Tkinter根窗口（不显示）
    root = tk.Tk()
    root.withdraw()  # 隐藏窗口

    # 创建画布
    canvas = tk.Canvas(root, width=280, height=280, bg='white')
    canvas.pack()

    # 在画布上绘制一些内容（模拟数字5）
    canvas.create_line(100, 50, 180, 50, width=10, fill='black')  # 顶部横线
    canvas.create_line(180, 50, 180, 150, width=10, fill='black')  # 右侧竖线
    canvas.create_line(180, 150, 100, 150, width=10, fill='black')  # 中间横线
    canvas.create_line(100, 150, 100, 250, width=10, fill='black')  # 左侧竖线
    canvas.create_line(100, 250, 180, 250, width=10, fill='black')  # 底部横线

    # 更新画布
    root.update()

    # 测试预处理
    print("调用预处理函数...")
    tensor = ImageProcessor.preprocess_canvas(canvas)

    if tensor is not None:
        print(f"预处理成功！张量形状: {tensor.shape}")
        print(f"张量值范围: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"张量均值: {tensor.mean():.4f}")

        # 显示一些统计信息
        print(f"张量非零元素比例: {(tensor > 0.1).sum().item() / tensor.numel():.4f}")

        return True
    else:
        print("预处理失败！")
        return False

    # 清理
    root.destroy()


def test_uploaded_image_preprocessing():
    """测试上传图像预处理。"""
    print("\n测试上传图像预处理...")

    # 创建一个测试图像
    from PIL import Image, ImageDraw
    import numpy as np

    # 创建28x28的测试图像（数字3）
    img = Image.new('L', (28, 28), color=255)  # 白色背景
    draw = ImageDraw.Draw(img)

    # 绘制数字3
    draw.line([(5, 5), (23, 5)], fill=0, width=2)  # 顶部横线
    draw.line([(23, 5), (23, 13)], fill=0, width=2)  # 右侧竖线
    draw.line([(23, 13), (5, 13)], fill=0, width=2)  # 中间横线
    draw.line([(23, 13), (23, 21)], fill=0, width=2)  # 右侧竖线
    draw.line([(23, 21), (5, 21)], fill=0, width=2)  # 底部横线

    # 保存测试图像
    test_image_path = "test_image.png"
    img.save(test_image_path)
    print(f"创建测试图像: {test_image_path}")

    # 测试预处理
    tensor = ImageProcessor.preprocess_uploaded_image(test_image_path)

    if tensor is not None:
        print(f"上传图像预处理成功！张量形状: {tensor.shape}")
        print(f"张量值范围: [{tensor.min():.4f}, {tensor.max():.4f}]")

        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

        return True
    else:
        print("上传图像预处理失败！")
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        return False


def main():
    """主测试函数。"""
    print("=" * 60)
    print("图像预处理模块测试")
    print("=" * 60)

    # 测试画布预处理
    success1 = test_canvas_preprocessing()

    # 测试上传图像预处理
    success2 = test_uploaded_image_preprocessing()

    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  画布预处理: {'成功' if success1 else '失败'}")
    print(f"  上传图像预处理: {'成功' if success2 else '失败'}")
    print("=" * 60)

    return success1 and success2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)