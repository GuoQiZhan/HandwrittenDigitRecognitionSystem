import numpy as np
from PIL import Image, ImageOps
import io
import tkinter as tk
from typing import Tuple, Optional
import torch


class ImageProcessor:
    """图像预处理模块，负责将手绘和上传的图像转换为模型输入格式。"""

    @staticmethod
    def preprocess_canvas(canvas: tk.Canvas, width: int = 280, height: int = 280) -> Optional[torch.Tensor]:
        """从Tkinter画布中获取图像并预处理。

        Args:
            canvas: Tkinter画布对象
            width: 画布宽度
            height: 画布高度

        Returns:
            预处理后的张量 (1, 1, 28, 28)，失败时返回None
        """
        try:
            # 首先确保画布已更新 - 更积极的更新策略
            for _ in range(3):  # 多次更新以确保画布完全刷新
                canvas.update()
                canvas.update_idletasks()

            # 方法1：使用PostScript（需要Ghostscript）
            try:
                # 在调用postscript前再次更新
                canvas.update()
                ps = canvas.postscript(colormode='gray')
                img = Image.open(io.BytesIO(ps.encode('utf-8')))
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                print("使用PostScript方法成功获取画布图像")
                method_used = "PostScript"
            except Exception as e:
                print(f"PostScript方法失败: {e}")
                print("尝试使用ImageGrab备用方法...")

                # 方法2：使用ImageGrab捕获画布区域
                from PIL import ImageGrab

                # 获取画布在屏幕上的位置，确保窗口已更新
                for _ in range(3):
                    canvas.update()
                    canvas.update_idletasks()

                # 确保窗口已映射到屏幕（可见）
                if not canvas.winfo_viewable():
                    print("警告: 画布不可见，强制更新窗口")
                    canvas.master.update()  # 更新父窗口

                # 获取屏幕坐标，确保坐标有效
                x = canvas.winfo_rootx()
                y = canvas.winfo_rooty()
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()

                # 验证坐标和尺寸是否合理
                if x < 0 or y < 0 or canvas_width <= 0 or canvas_height <= 0:
                    print(f"警告: 无效的画布坐标或尺寸: x={x}, y={y}, width={canvas_width}, height={canvas_height}")
                    # 使用画布配置的尺寸作为后备
                    if hasattr(canvas, 'width') and hasattr(canvas, 'height'):
                        canvas_width = canvas.width
                        canvas_height = canvas.height
                        # 使用窗口位置作为估计（可能不准确）
                        x = canvas.winfo_x() + canvas.master.winfo_rootx()
                        y = canvas.winfo_y() + canvas.master.winfo_rooty()
                        print(f"使用备用尺寸: {canvas_width}x{canvas_height}, 估计位置: ({x}, {y})")
                    else:
                        # 使用默认尺寸
                        canvas_width = width
                        canvas_height = height

                x2 = x + canvas_width
                y2 = y + canvas_height

                print(f"ImageGrab捕获区域: ({x}, {y}, {x2}, {y2}), 画布尺寸: {canvas_width}x{canvas_height}")

                # 捕获屏幕区域
                img = ImageGrab.grab(bbox=(x, y, x2, y2))
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                print(f"使用ImageGrab方法成功，捕获区域: ({x}, {y}, {x2}, {y2})")
                method_used = "ImageGrab"

        except Exception as e2:
            print(f"所有图像捕获方法都失败: {e2}")
            return None

        try:
            # 转换为灰度图
            img_gray = img.convert('L')

            # 转换为numpy数组
            img_array = np.array(img_gray, dtype=np.float32)

            print(f"原始图像统计 - 最小值: {img_array.min()}, 最大值: {img_array.max()}, 平均值: {img_array.mean():.2f}")

            # 亮度检查和自动反色
            # 画布应该是白底黑字，所以背景应该较亮（值接近255）
            mean_brightness = img_array.mean()
            print(f"图像平均亮度: {mean_brightness:.2f}")

            if mean_brightness < 127:
                # 图像整体较暗，可能是捕获了深色背景或反色了
                # 反色处理：确保背景为亮色，笔画为暗色
                print(f"图像较暗（平均亮度{mean_brightness:.2f} < 127），进行反色处理")
                img_array = 255 - img_array
                print(f"反色后统计 - 最小值: {img_array.min()}, 最大值: {img_array.max()}, 平均值: {img_array.mean():.2f}")
            else:
                print(f"图像亮度正常（平均亮度{mean_brightness:.2f} >= 127），保持原样")

            # 反色处理：画布上黑色笔画在白色背景上，需要转换为黑底白字
            # 画布背景为白色（255），笔画为黑色（0）或彩色
            # 我们通过阈值处理提取笔画
            # 将图像二值化：笔画区域为1，背景为0
            # 使用自适应阈值：基于图像亮度动态调整
            # 如果图像整体较亮，提高阈值；如果较暗，降低阈值
            dynamic_threshold = max(150, min(220, 255 - mean_brightness))
            print(f"使用动态阈值: {dynamic_threshold:.1f} (基于平均亮度{mean_brightness:.2f})")

            binary = np.where(img_array < dynamic_threshold, 1.0, 0.0)

            print(f"二值化后统计 - 笔画像素比例: {(binary > 0.5).sum() / binary.size:.4f}")

            # 验证是否有足够的笔画像素
            stroke_ratio = (binary > 0.5).sum() / binary.size
            if stroke_ratio < 0.001:  # 如果笔画像素比例太低，可能图像捕获失败
                print(f"警告: 笔画像素比例过低 ({stroke_ratio:.6f})，可能图像捕获失败或画布为空")
                # 检查原始图像是否全白或全黑
                if img_array.min() == img_array.max():
                    print(f"错误: 原始图像为单一颜色 ({img_array.min()})，图像捕获可能失败")
                return None

            # 调整大小为28x28
            # 将二值化数组转换回PIL图像进行resize
            binary_uint8 = (binary * 255).astype(np.uint8)
            binary_img = Image.fromarray(binary_uint8, mode='L')
            resized_img = binary_img.resize((28, 28), Image.Resampling.LANCZOS)
            resized = np.array(resized_img, dtype=np.float32) / 255.0

            print(f"调整大小后统计 - 最小值: {resized.min()}, 最大值: {resized.max()}, 平均值: {resized.mean():.4f}")

            # 转换为张量并添加批次和通道维度
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()

            print(f"预处理成功，使用方法: {method_used}, 张量形状: {tensor.shape}")
            return tensor

        except Exception as e:
            print(f"图像预处理过程中出错: {e}")
            return None

    @staticmethod
    def preprocess_uploaded_image(image_path: str) -> Optional[torch.Tensor]:
        """预处理上传的图像文件。

        Args:
            image_path: 图像文件路径

        Returns:
            预处理后的张量 (1, 1, 28, 28)，失败时返回None
        """
        try:
            # 打开图像
            img = Image.open(image_path)

            # 转换为灰度图
            img_gray = img.convert('L')

            # 调整大小为28x28
            img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

            # 转换为numpy数组
            img_array = np.array(img_resized, dtype=np.float32)

            # 反色处理：确保数字为白色，背景为黑色
            # 如果图像是白底黑字，需要反色
            # 通过检查图像的平均亮度来决定是否需要反色
            mean_brightness = np.mean(img_array)
            if mean_brightness > 127:
                # 图像整体较亮，可能是白底黑字，需要反色
                img_array = 255 - img_array

            # 归一化到[0, 1]范围
            img_normalized = img_array / 255.0

            # 转换为张量并添加批次和通道维度
            tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()

            return tensor

        except Exception as e:
            print(f"预处理上传图像时出错: {e}")
            return None

    @staticmethod
    def canvas_to_image(canvas: tk.Canvas, output_path: str, width: int = 280, height: int = 280) -> bool:
        """将画布内容保存为图像文件。

        Args:
            canvas: Tkinter画布对象
            output_path: 输出文件路径
            width: 输出图像宽度
            height: 输出图像高度

        Returns:
            成功标志
        """
        try:
            ps = canvas.postscript(colormode='gray')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            img.save(output_path)
            return True
        except Exception as e:
            print(f"保存画布图像时出错: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        tensor = ImageProcessor.preprocess_uploaded_image(sys.argv[1])
        if tensor is not None:
            print(f"预处理成功，张量形状: {tensor.shape}")
        else:
            print("预处理失败")
    else:
        print("请提供测试图像路径")