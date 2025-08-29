import os

root_dir = "/Users/zhenghaoyu/Downloads/hymenoptera_data/train"
target_dir = "ants_image"
img_path = os.path.join(root_dir, target_dir)
label = target_dir.split('_')[0]
out_dir = "ants_label"

# 创建输出目录
out_path = os.path.join(root_dir, out_dir)
os.makedirs(out_path, exist_ok=True)

# 遍历图片文件
for filename in os.listdir(img_path):
    if filename.endswith('.jpg'):
        file_name = filename.split('.jpg')[0]
        with open(os.path.join(out_path, f"{file_name}.txt"), 'w') as f:
            f.write(label)