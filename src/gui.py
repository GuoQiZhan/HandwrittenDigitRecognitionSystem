import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from typing import Optional
import threading

from .model_loader import ModelLoader
from .predictor import DigitPredictor
from .drawing_canvas import DrawingCanvas
from .image_processor import ImageProcessor


class DigitRecognizerGUI:
    """手写数字识别主GUI界面。"""

    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别系统")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # 初始化组件
        self.model_loader = ModelLoader()
        self.predictor = DigitPredictor(self.model_loader)

        # 当前选择的图片路径
        self.current_image_path: Optional[str] = None

        # 创建界面
        self.create_widgets()

        # 设置关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """创建所有界面组件。"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 1. 模型加载区域
        self.create_model_loading_area(main_frame)

        # 2. 手绘和上传区域（左右布局）
        drawing_frame = ttk.Frame(main_frame)
        drawing_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        drawing_frame.columnconfigure(0, weight=1)
        drawing_frame.columnconfigure(1, weight=1)
        drawing_frame.rowconfigure(0, weight=1)

        # 左侧：手绘画板区域
        self.create_drawing_area(drawing_frame)

        # 右侧：图片上传区域
        self.create_upload_area(drawing_frame)

        # 3. 识别结果区域
        self.create_result_area(main_frame)

        # 4. 状态栏
        self.create_status_bar(main_frame)

    def create_model_loading_area(self, parent):
        """创建模型加载区域。"""
        frame = ttk.LabelFrame(parent, text="模型加载", padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 模型路径显示
        path_frame = ttk.Frame(frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(path_frame, text="模型文件:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value="未选择模型文件")
        self.model_path_label = ttk.Label(path_frame, textvariable=self.model_path_var,
                                          foreground="gray", width=50)
        self.model_path_label.pack(side=tk.LEFT, padx=5)

        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="选择模型文件",
                   command=self.browse_model_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载模型",
                   command=self.load_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="卸载模型",
                   command=self.unload_model).pack(side=tk.LEFT, padx=2)

        # 模型状态
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(status_frame, text="模型状态:").pack(side=tk.LEFT)
        self.model_status_var = tk.StringVar(value="未加载")
        self.model_status_label = ttk.Label(status_frame, textvariable=self.model_status_var,
                                            foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=5)

        # 设备信息
        self.device_var = tk.StringVar(value="设备: CPU")
        ttk.Label(status_frame, textvariable=self.device_var).pack(side=tk.RIGHT)

    def create_drawing_area(self, parent):
        """创建手绘画板区域。"""
        frame = ttk.LabelFrame(parent, text="手绘画板", padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # 画板
        self.canvas = DrawingCanvas(frame, width=280, height=280)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # 控制按钮
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=1, column=0, pady=(10, 0))

        ttk.Button(control_frame, text="清空画板",
                   command=self.clear_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="识别手绘数字",
                   command=self.recognize_drawing).pack(side=tk.LEFT, padx=2)

        # 画笔设置
        brush_frame = ttk.Frame(frame)
        brush_frame.grid(row=2, column=0, pady=(5, 0))

        ttk.Label(brush_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_size_var = tk.IntVar(value=10)
        brush_spin = ttk.Spinbox(brush_frame, from_=1, to=30, width=5,
                                 textvariable=self.brush_size_var,
                                 command=self.update_brush_size)
        brush_spin.pack(side=tk.LEFT, padx=5)

    def create_upload_area(self, parent):
        """创建图片上传区域。"""
        frame = ttk.LabelFrame(parent, text="图片上传", padding="10")
        frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # 图片预览框架（固定尺寸）
        preview_frame = tk.Frame(frame, width=200, height=200, bg='white', relief=tk.SUNKEN)
        preview_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.grid_propagate(False)  # 禁止自动调整大小
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # 图片预览标签（放在框架中）
        self.image_preview_label = tk.Label(preview_frame, text="图片预览区域\n(支持PNG, JPG, BMP)",
                                            background="white",
                                            foreground="gray",
                                            font=("Arial", 10),
                                            anchor=tk.CENTER)
        self.image_preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 控制按钮
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=1, column=0, pady=(10, 0))

        ttk.Button(control_frame, text="选择图片",
                   command=self.browse_image_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="识别上传数字",
                   command=self.recognize_uploaded_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="清除图片",
                   command=self.clear_uploaded_image).pack(side=tk.LEFT, padx=2)

        # 图片信息
        self.image_info_var = tk.StringVar(value="未选择图片")
        ttk.Label(frame, textvariable=self.image_info_var).grid(row=2, column=0, pady=(5, 0))

    def create_result_area(self, parent):
        """创建识别结果区域。"""
        frame = ttk.LabelFrame(parent, text="识别结果", padding="10")
        frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # 主要结果
        result_frame = ttk.Frame(frame)
        result_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(result_frame, text="识别结果:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.result_var = tk.StringVar(value="")
        self.result_label = ttk.Label(result_frame, textvariable=self.result_var,
                                      font=("Arial", 24, "bold"),
                                      foreground="blue")
        self.result_label.pack(side=tk.LEFT, padx=20)

        # 置信度
        confidence_frame = ttk.Frame(frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(confidence_frame, text="置信度:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="")
        self.confidence_label = ttk.Label(confidence_frame, textvariable=self.confidence_var,
                                          font=("Arial", 12),
                                          foreground="green")
        self.confidence_label.pack(side=tk.LEFT, padx=20)

        # 详细信息（前3个结果）
        detail_frame = ttk.Frame(frame)
        detail_frame.pack(fill=tk.X)

        ttk.Label(detail_frame, text="详细信息:").pack(side=tk.LEFT)
        self.detail_var = tk.StringVar(value="")
        ttk.Label(detail_frame, textvariable=self.detail_var).pack(side=tk.LEFT, padx=20)

    def create_status_bar(self, parent):
        """创建状态栏。"""
        frame = ttk.Frame(parent, relief=tk.SUNKEN)
        frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ==================== 事件处理函数 ====================

    def browse_model_file(self):
        """浏览模型文件。"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型文件", "*.pth"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.set_status(f"已选择模型文件: {os.path.basename(file_path)}")

    def load_model(self):
        """加载模型。"""
        model_path = self.model_path_var.get()
        if not model_path or model_path == "未选择模型文件":
            messagebox.showwarning("警告", "请先选择模型文件")
            return

        # 在独立线程中加载模型，避免界面冻结
        def load_thread():
            self.set_status("正在加载模型...")
            success, msg = self.model_loader.load_model(model_path)
            if success:
                self.model_status_var.set("已加载")
                self.model_status_label.config(foreground="green")
                self.device_var.set(f"设备: {self.model_loader.get_device()}")
                self.set_status(f"模型加载成功: {msg}")
            else:
                self.model_status_var.set("加载失败")
                self.model_status_label.config(foreground="red")
                self.set_status(f"模型加载失败: {msg}")
                messagebox.showerror("加载失败", f"模型加载失败:\n{msg}")

        threading.Thread(target=load_thread, daemon=True).start()

    def unload_model(self):
        """卸载模型。"""
        self.model_loader.unload_model()
        self.model_status_var.set("未加载")
        self.model_status_label.config(foreground="red")
        self.set_status("模型已卸载")

    def clear_canvas(self):
        """清空画板。"""
        self.canvas.clear_canvas()
        self.set_status("画板已清空")

    def update_brush_size(self):
        """更新画笔大小。"""
        size = self.brush_size_var.get()
        self.canvas.set_brush_size(size)
        self.set_status(f"画笔大小已更新: {size}")

    def browse_image_file(self):
        """浏览图片文件。"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            file_name = os.path.basename(file_path)
            self.image_info_var.set(file_name)
            self.set_status(f"已选择图片: {file_name}")

            # 显示图片预览
            try:
                from PIL import Image, ImageTk
                img = Image.open(file_path)

                # 转换RGBA模式为RGB（如果图片有透明度）
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 创建一个白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        # 合并透明图层到白色背景
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else img)
                        img = background
                    else:
                        img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整图片大小以适应预览区域，保持宽高比
                preview_width, preview_height = 190, 190  # 略小于框架尺寸
                img_width, img_height = img.size

                # 计算缩放比例
                width_ratio = preview_width / img_width
                height_ratio = preview_height / img_height
                scale_ratio = min(width_ratio, height_ratio)

                if scale_ratio < 1:
                    new_size = (int(img_width * scale_ratio), int(img_height * scale_ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 创建PhotoImage并显示
                photo = ImageTk.PhotoImage(img)
                self.image_preview_label.config(image=photo, text="")
                self.image_preview_label.image = photo  # 保持引用

                self.set_status(f"图片预览已显示: {file_name}")

            except Exception as e:
                self.set_status(f"图片预览失败: {e}")
                print(f"图片预览错误: {e}")

    def clear_uploaded_image(self):
        """清除上传的图片。"""
        self.current_image_path = None
        self.image_info_var.set("未选择图片")
        self.image_preview_label.config(image="", text="图片预览区域\n(支持PNG, JPG, BMP)")
        if hasattr(self.image_preview_label, 'image'):
            self.image_preview_label.image = None
        self.set_status("已清除图片")

    def recognize_drawing(self):
        """识别手绘数字。"""
        if not self.model_loader.is_loaded():
            messagebox.showwarning("警告", "请先加载模型")
            return

        # 检查画板是否有内容
        # 简单检查：如果画板中只有边框和提示文字，则认为为空
        items = self.canvas.find_all()
        if len(items) <= 2:  # 边框和提示文字
            messagebox.showwarning("警告", "请在画板上绘制数字")
            return

        # 在独立线程中进行识别
        def recognize_thread():
            self.set_status("正在识别手绘数字...")
            success, results, error = self.predictor.predict_from_canvas(self.canvas)

            if success:
                self.show_results(results)
                self.set_status("手绘数字识别完成")
            else:
                self.clear_results()
                self.set_status(f"识别失败: {error}")
                messagebox.showerror("识别失败", f"手绘数字识别失败:\n{error}")

        threading.Thread(target=recognize_thread, daemon=True).start()

    def recognize_uploaded_image(self):
        """识别上传的图片。"""
        if not self.model_loader.is_loaded():
            messagebox.showwarning("警告", "请先加载模型")
            return

        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图片")
            return

        # 在独立线程中进行识别
        def recognize_thread():
            self.set_status("正在识别上传图片...")
            success, results, error = self.predictor.predict_from_image_file(self.current_image_path)

            if success:
                self.show_results(results)
                self.set_status("图片数字识别完成")
            else:
                self.clear_results()
                self.set_status(f"识别失败: {error}")
                messagebox.showerror("识别失败", f"图片数字识别失败:\n{error}")

        threading.Thread(target=recognize_thread, daemon=True).start()

    def show_results(self, results):
        """显示识别结果。"""
        if not results:
            self.clear_results()
            return

        # 主要结果（最高置信度）
        digit, confidence = results[0]
        self.result_var.set(str(digit))
        self.confidence_var.set(f"{confidence * 100:.2f}%")

        # 详细信息（前3个结果）
        details = []
        for i, (d, c) in enumerate(results):
            details.append(f"{i + 1}. 数字 {d}: {c * 100:.2f}%")
        self.detail_var.set(" | ".join(details))

        # 根据置信度设置颜色
        if confidence > 0.9:
            self.result_label.config(foreground="darkgreen")
            self.confidence_label.config(foreground="darkgreen")
        elif confidence > 0.7:
            self.result_label.config(foreground="blue")
            self.confidence_label.config(foreground="blue")
        else:
            self.result_label.config(foreground="orange")
            self.confidence_label.config(foreground="orange")

    def clear_results(self):
        """清除结果。"""
        self.result_var.set("")
        self.confidence_var.set("")
        self.detail_var.set("")
        self.result_label.config(foreground="blue")
        self.confidence_label.config(foreground="green")

    def set_status(self, message: str):
        """设置状态栏消息。"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_closing(self):
        """关闭窗口事件。"""
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()


def main():
    """启动GUI应用程序。"""
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()