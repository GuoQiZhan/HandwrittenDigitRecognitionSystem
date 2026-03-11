#!/usr/bin/env python3
"""
简单模型推理测试。
测试模型加载和基本预测功能。
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import ModelLoader
from src.predictor import DigitPredictor


def test_model_inference():
    """测试模型推理。"""
    print("模型推理测试")
    print("=" * 60)

    # 1. 加载模型
    model_path = "models/test_model.pth"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return False

    loader = ModelLoader()
    success, msg = loader.load_model(model_path)
    if not success:
        print(f"模型加载失败: {msg}")
        return False

    print(f"模型加载成功: {msg}")
    print(f"设备: {loader.get_device()}")

    # 2. 创建预测器
    predictor = DigitPredictor(loader)

    # 3. 创建测试张量（模拟数字5的特征）
    # 使用随机张量，但应该能产生某种预测
    test_tensor = torch.randn(1, 1, 28, 28)
    print(f"\n测试张量形状: {test_tensor.shape}")

    # 4. 进行预测
    success, results, error = predictor.predict(test_tensor, top_k=3)

    if not success:
        print(f"预测失败: {error}")
        return False

    # 5. 显示结果
    print("\n预测结果:")
    for i, (digit, confidence) in enumerate(results):
        print(f"  预测 {i+1}: 数字 {digit}, 置信度: {confidence*100:.2f}%")

    # 6. 检查结果是否合理
    # 置信度应该在0-1之间
    for digit, confidence in results:
        if confidence < 0 or confidence > 1:
            print(f"错误: 置信度超出范围: {confidence}")
            return False

    print("\n模型推理测试通过！")
    return True


def test_predictor_ready():
    """测试预测器就绪状态。"""
    print("\n预测器就绪状态测试")
    print("=" * 60)

    # 1. 未加载模型的情况
    loader = ModelLoader()
    predictor = DigitPredictor(loader)

    if predictor.is_ready():
        print("错误: 预测器在模型未加载时显示就绪")
        return False
    else:
        print("正确: 预测器在模型未加载时显示未就绪")

    # 2. 加载模型后的情况
    model_path = "models/test_model.pth"
    if os.path.exists(model_path):
        loader.load_model(model_path)
        predictor = DigitPredictor(loader)

        if predictor.is_ready():
            print("正确: 预测器在模型加载后显示就绪")
            return True
        else:
            print("错误: 预测器在模型加载后显示未就绪")
            return False
    else:
        print(f"模型文件不存在: {model_path}")
        return False


def main():
    """主测试函数。"""
    print("手写数字识别系统模型测试")
    print("=" * 60)

    # 测试1: 预测器就绪状态
    test1_passed = test_predictor_ready()

    # 测试2: 模型推理
    test2_passed = test_model_inference()

    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  预测器就绪状态测试: {'通过' if test1_passed else '失败'}")
    print(f"  模型推理测试: {'通过' if test2_passed else '失败'}")
    print("=" * 60)

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)