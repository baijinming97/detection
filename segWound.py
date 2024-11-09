from utils import *

num_interp = 10

# 定义SAM模型加载和预测的配置
sam_checkpoint = "D:\Projects\Wound\sam_vit_l_0b3195.pth"
model_type = "_".join(sam_checkpoint.split("_")[1:3])
device = "cuda"

# 加载SAM模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device)
predictor = SamPredictor(sam)

# 定义图片根目录
root_ = "D:\Projects\Wound\data"
paths = os.listdir(root_)

# 遍历根目录下的所有目录
for r in paths:
    root = os.path.join(root_, r)
    if not os.path.isdir(root):
        continue

    # 遍历每个目录下的子目录
    sub_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(root, sub_dir)

        # 查找所有的json文件
        json_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.json')]
        data = []

        # 处理每个json文件和对应的图片
        for i in tqdm(range(len(json_files))):
            json_file = os.path.join(sub_dir_path, json_files[i])
            image_file = json_file.replace(".json", ".tif")
            points = read_pts(json_file)
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_point = points
            input_label = np.array(np.ones(len(points)))
            predictor.set_image(image)
            mask, score, logit = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            mask[0] = largest_component(mask[0]).astype(bool)
            name = os.path.basename(image_file)
            rate = np.round(np.mean(mask[0]) * 100, 4)
            data.append([name, rate])
            color_mask = label_to_color_image(mask[0].astype(np.uint8))
            result = cv2.addWeighted(image, 1, color_mask, 0.1, 0.0)
            # for point in input_point:
            #     cv2.circle(result, tuple(point.astype(int)), 13, (255, 0, 0), -1)
            # im = show2d_image([image, result])
            cv2.imwrite(image_file.replace(".tif", ".png"), result)
            # im = im[:, :, ::-1]
            # target_width = int(im.shape[1] * 0.3)
            # target_height = int(im.shape[0] * 0.3)
            # resized_im = cv2.resize(im, (target_width, target_height))
            # cv2.imwrite(image_file.replace(".tif", ".png"), resized_im)

        # 保存结果到Excel文件
        df = pd.DataFrame(data, columns=["Name", "Rate"])
        output_file_name = f"{sub_dir}_{model_type}_output.xlsx"
        output_file_path = os.path.join(sub_dir_path, output_file_name)
        df.to_excel(output_file_path, index=False)

