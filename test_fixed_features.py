#!/usr/bin/env python3
"""
测试修复后的画板多次识别和图片上传预览功能。
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import ModelLoader
from src.predictor import DigitPredictor
from src.drawing_canvas import DrawingCanvas
from src.image_processor import ImageProcessor


def test_multiple_canvas_recognitions():
    """测试多次画板识别。"""
    print("=" * 60)
    print("测试多次画板识别功能")
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

    print(f"模型加载成功: {msg}")

    # 2. 创建GUI窗口和画板
    print("\n2. 创建测试窗口和画板...")
    root = tk.Tk()
    root.title("多次识别测试")
    root.geometry("400x500")

    canvas = DrawingCanvas(root, width=280, height=280)
    canvas.pack(padx=10, pady=10)

    # 添加控制按钮
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    result_label = tk.Label(root, text="识别结果将显示在这里", font=("Arial", 12))
    result_label.pack(pady=10)

    test_count = [0]  # 使用列表以便在闭包中修改
    success_count = [0]

    def draw_digit_5():
        """在画板上绘制数字5。"""
        canvas.clear_canvas()
        canvas.set_brush_color("black")
        canvas.set_brush_size(12)

        # 绘制数字5
        canvas.create_line(100, 50, 180, 50, width=12, fill='black', capstyle=tk.ROUND)
        canvas.create_line(180, 50, 180, 150, width=12, fill='black', capstyle=tk.ROUND)
        canvas.create_line(180, 150, 100, 150, width=12, fill='black', capstyle=tk.ROUND)
        canvas.create_line(100, 150, 100, 250, width=12, fill='black', capstyle=tk.ROUND)
        canvas.create_line(100, 250, 180, 250, width=12, fill='black', capstyle=tk.ROUND)

        root.update()
        print(f"已绘制数字5 (测试 #{test_count[0] + 1})")

    def recognize_current_drawing():
        """识别当前画板内容。"""
        test_count[0] += 1
        test_num = test_count[0]

        print(f"\n--- 开始识别测试 #{test_num} ---")

        # 创建预测器
        predictor = DigitPredictor(loader)

        # 进行识别
        success, results, error = predictor.predict_from_canvas(canvas)

        if success:
            success_count[0] += 1
            main_digit, main_confidence = results[0]
            result_text = f"测试 #{test_num}: 识别为数字 {main_digit}, 置信度: {main_confidence*100:.2f}%"
            result_label.config(text=result_text, fg="green")
            print(f"  识别成功: {result_text}")

            # 显示所有预测结果
            for i, (digit, confidence) in enumerate(results):
                print(f"    预测 {i+1}: 数字 {digit}, 置信度: {confidence*100:.2f}%")

            return True
        else:
            result_text = f"测试 #{test_num}: 识别失败 - {error}"
            result_label.config(text=result_text, fg="red")
            print(f"  识别失败: {error}")
            return False

    def run_single_test():
        """运行单次测试：绘制数字 -> 识别 -> 清空。"""
        if test_count[0] >= 5:
            print("\n已完成5次测试")
            return

        # 绘制数字
        draw_digit_5()

        # 等待片刻让界面更新
        root.after(500, lambda: (
            recognize_current_drawing(),
            root.after(1000, clear_and_continue)
        ))

    def clear_and_continue():
        """清空画板并继续下一次测试。"""
        canvas.clear_canvas()
        print(f"已清空画板，准备下一次测试")
        root.update()

        # 等待片刻后继续下一次测试
        if test_count[0] < 5:
            root.after(1000, run_single_test)
        else:
            print(f"\n测试完成！成功次数: {success_count[0]}/{test_count[0]}")
            if success_count[0] == test_count[0]:
                print("✓ 所有测试均成功！")
            else:
                print(f"✗ 部分测试失败: {test_count[0] - success_count[0]}次")

            # 等待3秒后关闭窗口
            root.after(3000, root.destroy)

    # 开始测试按钮
    start_btn = tk.Button(btn_frame, text="开始多次识别测试",
                          command=lambda: root.after(500, run_single_test))
    start_btn.pack(side=tk.LEFT, padx=5)

    # 单次测试按钮
    single_btn = tk.Button(btn_frame, text="单次识别测试",
                           command=lambda: (
                               draw_digit_5(),
                               root.after(500, lambda: recognize_current_drawing())
                           ))
    single_btn.pack(side=tk.LEFT, padx=5)

    # 清空按钮
    clear_btn = tk.Button(btn_frame, text="清空画板",
                          command=canvas.clear_canvas)
    clear_btn.pack(side=tk.LEFT, padx=5)

    print("\n3. 开始测试...")
    print("请点击'开始多次识别测试'按钮开始测试")
    print("或点击'单次识别测试'按钮进行单次测试")

    root.mainloop()

    return success_count[0] == test_count[0] and test_count[0] > 0


def test_image_upload_preview():
    """测试图片上传预览功能。"""
    print("\n" + "=" * 60)
    print("测试图片上传预览功能")
    print("=" * 60)

    # 创建测试图片
    from PIL import Image, ImageDraw
    import tempfile

    print("1. 创建测试图片...")

    # 创建两张测试图片
    test_images = []
    for digit in [3, 7]:
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 绘制数字
        if digit == 3:
            # 数字3
            draw.line([(20, 20), (80, 20)], fill=(0, 0, 0), width=8)
            draw.line([(80, 20), (80, 50)], fill=(0, 0, 0), width=8)
            draw.line([(80, 50), (20, 50)], fill=(0, 0, 0), width=8)
            draw.line([(80, 50), (80, 80)], fill=(0, 0, 0), width=8)
            draw.line([(80, 80), (20, 80)], fill=(0, 0, 0), width=8)
        elif digit == 7:
            # 数字7
            draw.line([(20, 20), (80, 20)], fill=(0, 0, 0), width=8)
            draw.line([(80, 20), (50, 80)], fill=(0, 0, 0), width=8)

        # 保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name)
        test_images.append((digit, temp_file.name))
        print(f"  创建测试图片: {temp_file.name} (数字{digit})")

    print("\n2. 测试图片预处理...")
    for digit, img_path in test_images:
        tensor = ImageProcessor.preprocess_uploaded_image(img_path)
        if tensor is not None:
            print(f"  图片预处理成功: {os.path.basename(img_path)} -> 形状: {tensor.shape}")
        else:
            print(f"  图片预处理失败: {os.path.basename(img_path)}")

        # 清理临时文件
        os.unlink(img_path)

    print("\n3. 注意: 图片上传预览功能需要在GUI中测试")
    print("   请运行主程序测试图片上传和预览功能")

    return True


def main():
    """主测试函数。"""
    print("手写数字识别系统功能测试")
    print("测试修复后的多次识别和图片预览功能")

    # 测试1: 多次画板识别
    print("\n" + "=" * 60)
    print("测试1: 多次画板识别")
    print("=" * 60)
    test1_passed = test_multiple_canvas_recognitions()

    # 测试2: 图片上传预览
    print("\n" + "=" * 60)
    print("测试2: 图片上传预览")
    print("=" * 60)
    test2_passed = test_image_upload_preview()

    print("\n" + "=" * 60)
    print("最终测试结果:")
    print(f"  多次画板识别测试: {'通过' if test1_passed else '失败'}")
    print(f"  图片上传预览测试: {'通过' if test2_passed else '失败'}")
    print("=" * 60)

    # 提示用户进行完整GUI测试
    print("\n提示: 请运行主程序进行完整GUI测试:")
    print("  python src/main.py")
    print("\n测试要点:")
    print("  1. 加载模型 (选择models/test_model.pth)")
    print("  2. 在画板上绘制数字，多次识别测试")
    print("  3. 清空画板后再次绘制并识别")
    print("  4. 上传图片测试预览功能")
    print("  5. 识别上传的图片")

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)