import tkinter as tk
from typing import Optional, Tuple


class DrawingCanvas(tk.Canvas):
    """自定义手绘画板组件。"""

    def __init__(self, parent, width: int = 280, height: int = 280, bg_color: str = "white",
                 brush_color: str = "black", brush_size: int = 10):
        """初始化画板。

        Args:
            parent: 父容器
            width: 画板宽度
            height: 画板高度
            bg_color: 背景颜色
            brush_color: 画笔颜色
            brush_size: 画笔大小
        """
        super().__init__(parent, width=width, height=height, bg=bg_color,
                         highlightbackground="gray", highlightthickness=2)

        self.width = width
        self.height = height
        self.brush_color = brush_color
        self.brush_size = brush_size
        self.bg_color = bg_color

        # 绘图状态
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        self.drawing = False

        # 绑定鼠标事件
        self.bind("<Button-1>", self.start_drawing)
        self.bind("<B1-Motion>", self.draw)
        self.bind("<ButtonRelease-1>", self.stop_drawing)

        # 绘制初始边框和提示
        self.create_rectangle(5, 5, width - 5, height - 5, outline="lightgray", width=1)
        self.create_text(width // 2, height // 2, text="在此绘制数字", fill="lightgray", font=("Arial", 12))

    def start_drawing(self, event):
        """开始绘制。"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        # 移除提示文字
        self.delete("prompt")

    def draw(self, event):
        """绘制线条。"""
        if self.drawing and self.last_x is not None and self.last_y is not None:
            x, y = event.x, event.y
            # 绘制线条
            self.create_line(self.last_x, self.last_y, x, y,
                             fill=self.brush_color,
                             width=self.brush_size,
                             capstyle=tk.ROUND,
                             smooth=tk.TRUE)
            # 绘制点（确保单点也能显示）
            self.create_oval(x - self.brush_size // 2, y - self.brush_size // 2,
                             x + self.brush_size // 2, y + self.brush_size // 2,
                             fill=self.brush_color, outline=self.brush_color)

            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        """停止绘制。"""
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """清空画板。"""
        self.delete("all")
        # 重新绘制边框和提示
        self.create_rectangle(5, 5, self.width - 5, self.height - 5, outline="lightgray", width=1)
        self.create_text(self.width // 2, self.height // 2,
                         text="在此绘制数字", fill="lightgray", font=("Arial", 12), tags="prompt")
        self.last_x = None
        self.last_y = None
        self.drawing = False

    def get_canvas_image(self) -> Optional[tk.PhotoImage]:
        """获取画板图像（仅用于预览，不用于识别）。"""
        try:
            # 使用postscript生成图像
            from PIL import Image, ImageTk
            import io

            ps = self.postscript(colormode='color')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img = img.resize((self.width, self.height))
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def get_canvas_size(self) -> Tuple[int, int]:
        """获取画板尺寸。"""
        return self.width, self.height

    def set_brush_color(self, color: str):
        """设置画笔颜色。"""
        self.brush_color = color

    def set_brush_size(self, size: int):
        """设置画笔大小。"""
        self.brush_size = max(1, min(size, 50))  # 限制范围


if __name__ == "__main__":
    # 测试代码
    root = tk.Tk()
    root.title("画板测试")

    canvas = DrawingCanvas(root)
    canvas.pack(padx=10, pady=10)

    # 控制按钮
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

    clear_btn = tk.Button(btn_frame, text="清空画板", command=canvas.clear_canvas)
    clear_btn.pack(side=tk.LEFT, padx=5)

    color_btn = tk.Button(btn_frame, text="红色画笔", command=lambda: canvas.set_brush_color("red"))
    color_btn.pack(side=tk.LEFT, padx=5)

    size_btn = tk.Button(btn_frame, text="画笔加大", command=lambda: canvas.set_brush_size(canvas.brush_size + 2))
    size_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()