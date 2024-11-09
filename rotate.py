from PIL import Image
import os

# 指定需要遍历的根文件夹路径
root_folder = 'D:\Projects\Wound\data'

# 使用os.walk()遍历文件夹及其所有子文件夹
for subdir, dirs, files in os.walk(root_folder):
    for filename in files:
        if filename.endswith('.tif'):
            # 构建完整的文件路径
            file_path = os.path.join(subdir, filename)
            
            # 打开图像
            img = Image.open(file_path)
            
            # 旋转图像
            img_rotated = img.rotate(90, expand=True)
            
            # 保存旋转后的图像，可以选择覆盖原文件或保存为新文件
            img_rotated.save(file_path)  # 如果要保存为新文件，可以修改文件名

print("所有图像处理完毕。")

