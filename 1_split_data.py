import os
import shutil
import random
from tqdm import tqdm
import yaml

# === CONFIG ===
input_folders = [
    r"real_dataset_Nerf_Bluegun",
    r"synthetic_dataset_Nerf_Bluegun",
    r"coco"
]
output_folder = r"dataset_Nerf_Bluegun_used"
train_part = 0.8
test_part = 1 - train_part

# === Combine all images and labels ===
image_paths = []
for folder in input_folders:
    img_dir = os.path.join(folder, "dataset", "images")
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(img_dir, img_name))

# === Shuffle and split ===
random.seed(42)
random.shuffle(image_paths)

split_idx = int(len(image_paths) * train_part)
train_images = image_paths[:split_idx]
test_images = image_paths[split_idx:]

# === Helper function to copy ===
def copy_data(img_list, split_name):
    for img_path in tqdm(img_list, desc=f"Copying {split_name} set"):
        label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        dst_img = os.path.join(output_folder, split_name, "images", os.path.basename(img_path))
        dst_lbl = os.path.join(output_folder, split_name, "labels", os.path.basename(label_path))
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        shutil.copy2(img_path, dst_img)
        if os.path.exists(label_path):
            shutil.copy2(label_path, dst_lbl)

copy_data(train_images, "train")
copy_data(test_images, "val")

# === Create data.yaml ===
data_yaml = {
    "train": "train",
    "val": "val",
    "nc": 1,  # number of classes
    "names": ["KAC_PDW_Blackgun"]  # adjust if you have more classes
}

yaml_path = os.path.join(output_folder, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"\n✅ Split complete! data.yaml saved at:\n{yaml_path}")
