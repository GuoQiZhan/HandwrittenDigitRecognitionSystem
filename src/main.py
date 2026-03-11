#!/usr/bin/env python3
"""
手写数字识别系统主程序入口。
"""

import sys
import os
import traceback
from tkinter import messagebox

# 添加项目根目录到Python路径，确保模块导入正常
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """检查必要的依赖包。"""
    required_packages = ['torch', 'PIL', 'numpy', 'matplotlib']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def main():
    """主函数。"""
    try:
        # 检查依赖
        missing = check_dependencies()
        if missing:
            error_msg = f"缺少必要的依赖包: {', '.join(missing)}\n\n"
            error_msg += "请安装依赖:\n"
            error_msg += "pip install torch torchvision Pillow numpy matplotlib"

            # 尝试在控制台显示错误
            print(error_msg, file=sys.stderr)

            # 如果可能，显示GUI错误提示
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("依赖缺失", error_msg)
                root.destroy()
            except:
                pass

            return 1

        # 导入GUI模块
        from src.gui import main as gui_main

        # 启动GUI
        gui_main()

    except Exception as e:
        # 捕获所有未处理的异常
        error_msg = f"程序启动失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)

        # 尝试显示错误对话框
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("程序错误", f"程序启动失败:\n{str(e)}")
            root.destroy()
        except:
            pass

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())