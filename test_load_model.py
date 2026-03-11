#!/usr/bin/env python3
"""
测试模型加载功能。
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import ModelLoader


def test_model_loading(model_path):
    """测试模型加载。"""
    print(f"测试加载模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"文件不存在: {model_path}")
        return False

    # 直接使用torch.load检查文件内容
    print("使用torch.load直接加载文件...")
    try:
        loaded = torch.load(model_path, map_location='cpu')
        print(f"加载的对象类型: {type(loaded)}")
        print(f"对象详细信息: {loaded}")

        if isinstance(loaded, list):
            print("加载的对象是列表，可能是保存了多个对象")
            if len(loaded) > 0:
                print(f"列表第一个元素类型: {type(loaded[0])}")
        elif isinstance(loaded, dict):
            print("加载的对象是字典（状态字典）")
            print(f"字典键: {list(loaded.keys())[:5]}...")
        elif isinstance(loaded, torch.nn.Module):
            print("加载的对象是完整的模型")
        else:
            print(f"未知对象类型: {type(loaded)}")

    except Exception as e:
        print(f"加载失败: {e}")
        return False

    # 使用ModelLoader加载
    print("\n使用ModelLoader加载...")
    loader = ModelLoader()
    success, msg = loader.load_model(model_path)
    print(f"加载结果: {success}, 消息: {msg}")

    if success:
        model = loader.get_model()
        print(f"模型类型: {type(model)}")
        print(f"模型设备: {loader.get_device()}")
        print(f"模型是否加载: {loader.is_loaded()}")

    return success


def main():
    """主测试函数。"""
    # 测试两个模型文件
    model_files = [
        "models/test_model.pth",
        "models/test_model_full.pth"
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            print("=" * 60)
            success = test_model_loading(model_file)
            if success:
                print(f"OK {model_file} 加载成功")
            else:
                print(f"FAIL {model_file} 加载失败")
            print("=" * 60)
        else:
            print(f"文件不存在: {model_file}")


if __name__ == "__main__":
    main()