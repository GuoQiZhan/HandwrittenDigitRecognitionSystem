#!/usr/bin/env python3
"""
手绘数字识别端到端测试。
测试完整流程：加载模型 -> 手绘数字 -> 识别 -> 显示结果。
"""

import sys
import os
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import ModelLoader
from src.predictor import DigitPredictor
from src.drawing_canvas import DrawingCanvas


def test_end_to_end():
    """端到端测试。"""
    print("=" * 60)
    print("手绘数字识别端到端测试")
    print("=" * 60)

    # 1. 加载模型
    print("1. 加载模型...")
    model_path = "models/test_model.pth"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train_mnist.py 训练模型")
        return False

    loader = ModelLoader()
    success, msg = loader.load_model(model_path)
    if not success:
        print(f"模型加载失败: {msg}")
        return False

    print(f"模型加载成功: {msg}")
    print(f"设备: {loader.get_device()}")

    # 2. 创建GUI窗口和画板
    print("\n2. 创建画板...")
    root = tk.Tk()
    root.title("测试窗口")
    root.geometry("400x400")

    canvas = DrawingCanvas(root, width=280, height=280)
    canvas.pack(padx=10, pady=10)

    # 3. 绘制测试数字（数字5）
    print("3. 在画板上绘制数字5...")
    # 使用画板API绘制
    canvas.set_brush_color("black")
    canvas.set_brush_size(12)

    # 模拟鼠标事件绘制数字5
    # 由于DrawingCanvas需要实际的鼠标事件，我们直接使用create_line
    # 顶部横线
    canvas.create_line(100, 50, 180, 50, width=12, fill='black', capstyle=tk.ROUND)
    # 右侧竖线
    canvas.create_line(180, 50, 180, 150, width=12, fill='black', capstyle=tk.ROUND)
    # 中间横线
    canvas.create_line(180, 150, 100, 150, width=12, fill='black', capstyle=tk.ROUND)
    # 左侧竖线
    canvas.create_line(100, 150, 100, 250, width=12, fill='black', capstyle=tk.ROUND)
    # 底部横线
    canvas.create_line(100, 250, 180, 250, width=12, fill='black', capstyle=tk.ROUND)

    # 更新窗口
    root.update()
    print("绘制完成，窗口已更新")

    # 4. 创建预测器并进行识别
    print("\n4. 进行数字识别...")
    predictor = DigitPredictor(loader)

    # 直接使用画板进行预测（会调用预处理和模型推理）
    success, results, error = predictor.predict_from_canvas(canvas)

    if not success:
        print(f"识别失败: {error}")
        root.destroy()
        return False

    # 5. 显示结果
    print("\n5. 识别结果:")
    for i, (digit, confidence) in enumerate(results):
        print(f"  预测 {i+1}: 数字 {digit}, 置信度: {confidence*100:.2f}%")

    # 检查主要预测结果
    main_digit, main_confidence = results[0]
    print(f"\n主要预测: 数字 {main_digit}, 置信度: {main_confidence*100:.2f}%")

    # 6. 验证结果（期望是数字5）
    expected_digit = 5
    if main_digit == expected_digit:
        print(f"测试通过！正确识别数字 {expected_digit}")
        test_passed = True
    else:
        print(f"测试失败！期望数字 {expected_digit}，但识别为 {main_digit}")
        test_passed = False

    # 7. 保存画布图像以供检查
    print("\n6. 保存画布图像...")
    output_path = "test_drawing.png"
    from src.image_processor import ImageProcessor
    if ImageProcessor.canvas_to_image(canvas, output_path):
        print(f"画布图像已保存: {output_path}")
    else:
        print("保存画布图像失败")

    # 清理
    root.destroy()
    print("\n测试完成！")

    return test_passed


def test_uploaded_image():
    """测试上传图片识别。"""
    print("\n" + "=" * 60)
    print("上传图片识别测试")
    print("=" * 60)

    # 1. 加载模型
    print("1. 加载模型...")
    model_path = "models/test_model.pth"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return False

    loader = ModelLoader()
    success, msg = loader.load_model(model_path)
    if not success:
        print(f"模型加载失败: {msg}")
        return False

    # 2. 创建测试图像（数字3）
    print("\n2. 创建测试图像（数字3）...")
    img = Image.new('L', (100, 100), color=255)  # 白色背景
    draw = ImageDraw.Draw(img)

    # 绘制数字3
    draw.line([(20, 20), (80, 20)], fill=0, width=8)  # 顶部横线
    draw.line([(80, 20), (80, 50)], fill=0, width=8)  # 右侧竖线
    draw.line([(80, 50), (20, 50)], fill=0, width=8)  # 中间横线
    draw.line([(80, 50), (80, 80)], fill=0, width=8)  # 右侧竖线
    draw.line([(80, 80), (20, 80)], fill=0, width=8)  # 底部横线

    test_image_path = "test_upload_3.png"
    img.save(test_image_path)
    print(f"测试图像已保存: {test_image_path}")

    # 3. 进行识别
    print("\n3. 进行图片识别...")
    predictor = DigitPredictor(loader)
    success, results, error = predictor.predict_from_image_file(test_image_path)

    if not success:
        print(f"识别失败: {error}")
        os.remove(test_image_path)
        return False

    # 4. 显示结果
    print("\n4. 识别结果:")
    for i, (digit, confidence) in enumerate(results):
        print(f"  预测 {i+1}: 数字 {digit}, 置信度: {confidence*100:.2f}%")

    main_digit, main_confidence = results[0]
    print(f"\n主要预测: 数字 {main_digit}, 置信度: {main_confidence*100:.2f}%")

    # 5. 验证结果（期望是数字3）
    expected_digit = 3
    if main_digit == expected_digit:
        print(f"测试通过！正确识别数字 {expected_digit}")
        test_passed = True
    else:
        print(f"测试失败！期望数字 {expected_digit}，但识别为 {main_digit}")
        test_passed = False

    # 清理
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    return test_passed


def main():
    """主测试函数。"""
    print("手写数字识别系统完整测试")
    print("注意: 测试期间会显示GUI窗口，请勿关闭")

    # 测试1: 手绘数字识别
    test1_passed = test_end_to_end()

    # 测试2: 上传图片识别
    test2_passed = test_uploaded_image()

    print("\n" + "=" * 60)
    print("最终测试结果:")
    print(f"  手绘数字识别测试: {'通过' if test1_passed else '失败'}")
    print(f"  上传图片识别测试: {'通过' if test2_passed else '失败'}")
    print("=" * 60)

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)