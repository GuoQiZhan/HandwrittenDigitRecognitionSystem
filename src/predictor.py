import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class DigitPredictor:
    """数字预测器，使用加载的模型进行预测。"""

    def __init__(self, model_loader):
        """初始化预测器。

        Args:
            model_loader: ModelLoader实例
        """
        self.model_loader = model_loader
        self.model = None
        self.device = torch.device('cpu')

    def is_ready(self) -> bool:
        """检查预测器是否准备好（模型已加载）。"""
        if self.model_loader.is_loaded():
            self.model = self.model_loader.get_model()
            self.device = self.model_loader.get_device()
            return True
        return False

    def predict(self, image_tensor: torch.Tensor, top_k: int = 3) -> Tuple[bool, List[Tuple[int, float]], Optional[str]]:
        """对输入图像进行预测。

        Args:
            image_tensor: 输入张量 (1, 1, 28, 28)
            top_k: 返回前K个预测结果

        Returns:
            (成功标志, [(数字, 置信度), ...], 错误信息)
        """
        if not self.is_ready():
            return False, [], "模型未加载"

        if image_tensor is None:
            return False, [], "输入图像为空"

        try:
            # 确保张量在正确的设备上
            image_tensor = image_tensor.to(self.device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

            # 转换为Python列表
            results = []
            for i in range(top_k):
                digit = int(top_indices[0, i].item())
                confidence = float(top_probs[0, i].item())
                results.append((digit, confidence))

            return True, results, None

        except Exception as e:
            error_msg = f"预测过程中出错: {str(e)}"
            return False, [], error_msg

    def predict_from_canvas(self, canvas, top_k: int = 3) -> Tuple[bool, List[Tuple[int, float]], Optional[str]]:
        """从画布直接预测。

        Args:
            canvas: Tkinter画布对象
            top_k: 返回前K个预测结果

        Returns:
            (成功标志, [(数字, 置信度), ...], 错误信息)
        """
        from .image_processor import ImageProcessor
        image_tensor = ImageProcessor.preprocess_canvas(canvas)
        if image_tensor is None:
            return False, [], "无法从画布预处理图像"
        return self.predict(image_tensor, top_k)

    def predict_from_image_file(self, image_path: str, top_k: int = 3) -> Tuple[bool, List[Tuple[int, float]], Optional[str]]:
        """从图像文件直接预测。

        Args:
            image_path: 图像文件路径
            top_k: 返回前K个预测结果

        Returns:
            (成功标志, [(数字, 置信度), ...], 错误信息)
        """
        from .image_processor import ImageProcessor
        image_tensor = ImageProcessor.preprocess_uploaded_image(image_path)
        if image_tensor is None:
            return False, [], "无法从图像文件预处理图像"
        return self.predict(image_tensor, top_k)

    def get_model_info(self) -> str:
        """获取模型信息。"""
        if self.model is not None:
            return f"模型架构: {self.model.__class__.__name__}, 设备: {self.device}"
        return "模型未加载"


if __name__ == "__main__":
    # 测试代码
    from model_loader import ModelLoader

    # 创建模型加载器
    loader = ModelLoader()

    # 测试模型加载（需要实际模型文件）
    test_model_path = "models/mnist_model.pth"
    import os
    if os.path.exists(test_model_path):
        success, msg = loader.load_model(test_model_path)
        print(f"模型加载: {success}, {msg}")

        if success:
            predictor = DigitPredictor(loader)

            # 创建随机测试张量
            test_tensor = torch.randn(1, 1, 28, 28)
            success, results, error = predictor.predict(test_tensor)

            if success:
                print("预测成功:")
                for digit, confidence in results:
                    print(f"  数字: {digit}, 置信度: {confidence:.4f}")
            else:
                print(f"预测失败: {error}")
    else:
        print(f"测试模型文件不存在: {test_model_path}")