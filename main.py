from utils import *
from tqdm import tqdm
import os
import cv2
import numpy as np
import pandas as pd

# 加载 YOLO 模型
model = YOLO('WoundYoloSeg.pt', verbose=False)

# 定义根目录
root = "D:\Projects\Wound\data"

# 列出所有 TIFF 文件
tif_files = list_tif_paths(root)

# 处理每个 TIFF 文件
with tqdm(tif_files, desc="Processing TIFF files") as bar:
    for impath in bar:
        # 确保文件路径中的空格被正确处理
        string_path = impath.replace("\\", "/")
        pnames = string_path.split("/")
        bar.set_description(f"{pnames[-3]}/{pnames[-2]}/{pnames[-1]}")

        # 检查文件路径是否存在
        if not os.path.isfile(impath):
            print(f"File not found: {impath}, skipping...")
            continue

        # 读取图像
        image = cv2.imread(impath)
        if image is None:
            print(f"Failed to read image {impath}, skipping...")
            continue

        h, w = image.shape[:2]
        extract = model([impath])[0].masks
        if extract is not None:
            mask = extract.data.cpu().numpy()[0].astype(np.uint8)
            mask = mask * 255
            mask = cv2.resize(mask, (w, h))
            create_json(mask, impath, h, w, erosion_size=5, grid_spacing=100)

# SAM 模型配置和预测
num_interp = 10
sam_checkpoint = "D:\Projects\Wound\sam_vit_l_0b3195.pth"
model_type = "_".join(sam_checkpoint.split("_")[1:3])
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device)
predictor = SamPredictor(sam)

root_ = "D:\Projects\Wound\data"
paths = os.listdir(root_)

# 处理每个目录
for r in paths:
    root = os.path.join(root_, r)
    if not os.path.isdir(root):
        continue
    sub_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(root, sub_dir)
        json_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.json')]
        data = []

        with tqdm(json_files, desc="Processing JSON files") as bar:
            for jfile in bar:
                json_file = os.path.join(sub_dir_path, jfile)
                string_path = json_file.replace("\\", "/")
                pnames = string_path.split("/")
                bar.set_description(f"{pnames[-3]}/{pnames[-2]}/{pnames[-1]}")
                image_file = json_file.replace(".json", ".tif")
                points = read_pts(json_file)

                if len(points) == 0:
                    print(f"No points found in {json_file}, skipping...")
                    continue

                image = cv2.imread(image_file)
                if image is None:
                    print(f"Failed to read image {image_file}, skipping...")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_point = points
                input_label = np.array(np.ones(len(points)))
                predictor.set_image(image)

                try:
                    mask, score, logit = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue

                mask[0] = largest_component(mask[0]).astype(bool)
                name = os.path.basename(image_file)
                rate = np.round(np.mean(mask[0]) * 100, 4)
                data.append([name, rate])
                color_mask = label_to_color_image(mask[0].astype(np.uint8))
                result = cv2.addWeighted(image, 1, color_mask, 0.1, 0.0)
                cv2.imwrite(image_file.replace(".tif", ".png"), result)

        # 保存结果到 Excel 文件
        df = pd.DataFrame(data, columns=["Name", "Rate"])
        output_file_name = f"{sub_dir}_{model_type}_output.xlsx"
        output_file_path = os.path.join(sub_dir_path, output_file_name)
        df.to_excel(output_file_path, index=False)

