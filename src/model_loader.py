import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Any


class MNISTCNN(nn.Module):
    """简单的CNN模型，用于MNIST手写数字识别。
    输入: 1x28x28 灰度图像
    输出: 10个类别 (0-9)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 第一次池化: 28x28 -> 14x14
        x = self.pool(x)  # 第二次池化: 14x14 -> 7x7
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ModelLoader:
    """模型加载器，负责加载.pth格式的模型文件并提供状态管理。"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded = False
        self.error_message = None

    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """加载模型文件。

        Args:
            model_path: .pth模型文件路径

        Returns:
            (成功标志, 消息字符串)
        """
        self.model_path = model_path
        self.loaded = False
        self.error_message = None

        # 检查文件是否存在
        if not os.path.exists(model_path):
            self.error_message = f"模型文件不存在: {model_path}"
            return False, self.error_message

        if not model_path.lower().endswith('.pth'):
            self.error_message = "模型文件必须是.pth格式"
            return False, self.error_message

        try:
            # 尝试加载整个模型
            model = torch.load(model_path, map_location=self.device)

            if isinstance(model, nn.Module):
                # 加载的是完整的模型对象
                self.model = model
                self.model.to(self.device)
                self.model.eval()
                self.loaded = True
                return True, "模型加载成功（完整模型对象）"
            elif isinstance(model, dict):
                # 加载的是状态字典，需要创建模型架构
                # 尝试使用预定义的MNISTCNN架构
                try:
                    self.model = MNISTCNN()
                    self.model.load_state_dict(model)
                    self.model.to(self.device)
                    self.model.eval()
                    self.loaded = True
                    return True, "模型加载成功（状态字典）"
                except Exception as e:
                    self.error_message = f"状态字典与模型架构不匹配: {str(e)}"
                    return False, self.error_message
            elif isinstance(model, list):
                # 加载的是列表，尝试查找模型或状态字典
                self.error_message = "加载的对象是列表格式，尝试查找模型或状态字典..."
                for i, item in enumerate(model):
                    if isinstance(item, nn.Module):
                        # 列表中的模型对象
                        self.model = item
                        self.model.to(self.device)
                        self.model.eval()
                        self.loaded = True
                        return True, f"模型加载成功（列表中的模型对象，索引{i}）"
                    elif isinstance(item, dict):
                        # 列表中的状态字典
                        try:
                            self.model = MNISTCNN()
                            self.model.load_state_dict(item)
                            self.model.to(self.device)
                            self.model.eval()
                            self.loaded = True
                            return True, f"模型加载成功（列表中的状态字典，索引{i}）"
                        except Exception as e:
                            self.error_message = f"列表中的状态字典与模型架构不匹配: {str(e)}"
                            continue
                self.error_message = f"列表中未找到有效的模型或状态字典，列表长度: {len(model)}"
                return False, self.error_message
            else:
                self.error_message = f"未知的模型格式: {type(model)}"
                return False, self.error_message

        except Exception as e:
            self.error_message = f"加载模型时出错: {str(e)}"
            return False, self.error_message

    def get_model(self) -> Optional[nn.Module]:
        """获取已加载的模型。"""
        return self.model if self.loaded else None

    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self.loaded

    def get_device(self) -> torch.device:
        """获取当前设备（CPU/CUDA）。"""
        return self.device

    def get_error_message(self) -> Optional[str]:
        """获取错误信息。"""
        return self.error_message

    def unload_model(self):
        """卸载当前模型。"""
        self.model = None
        self.loaded = False
        self.error_message = None


if __name__ == "__main__":
    # 测试代码
    loader = ModelLoader()
    test_path = "models/mnist_model.pth"
    if os.path.exists(test_path):
        success, msg = loader.load_model(test_path)
        print(f"加载结果: {success}, 消息: {msg}")
        if success:
            print(f"模型设备: {loader.get_device()}")
    else:
        print(f"测试模型文件不存在: {test_path}")