import os
from tkinter import Tk, Button, Label
from PIL import Image, ImageTk

# 图片展示应用程序
class ImageViewer:
    def __init__(self,image_paths):
        self.root = Tk()
        self.image_paths = image_paths
        self.index = 0

        # 设置窗口标题和尺寸
        self.root.title("图片查看器")
        self.root.geometry("800x700")

        # 图片显示区域
        self.label = Label(self.root)
        self.label.pack(expand=True, fill="both")

        # 按钮
        self.next_button = Button(self.root, text="下一张", command=self.next_image)
        self.next_button.pack(side="bottom")

        # 显示第一张图片
        self.show_image(self.index)

    def show_image(self, index):
        """显示当前索引的图片"""
        img_path = self.image_paths[index]
        image = Image.open(img_path)

        # 调整图片大小以适应窗口
        image.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo

    def next_image(self):
        """加载下一张图片"""
        self.index = (self.index + 1) % len(self.image_paths)  # 循环显示
        self.show_image(self.index)

    def __call__(self):
        # 启动应用
        self.root.mainloop()

if __name__ == "__main__":
    # 图片目录（替换为你实际的路径）
    image_dir = "./images"

    # 获取目录中的图片列表
    image_files = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    if not image_files:
        print("未找到任何图片！请确认图片路径。")
    else:
        viewer = ImageViewer(image_files)
        viewer()
