import os
import shutil
import random
from pathlib import Path
import yaml

def split_yolo_dataset(source_root, output_root, train_ratio=0.8):
    source_root = Path(source_root)
    output_root = Path(output_root)
    
    for split in ["Train", "Test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    dataset_root = source_root / "dataset"
    folders = [f for f in dataset_root.iterdir() if f.is_dir()]

    for folder in folders:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        images.sort()
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        train_imgs = images[:n_train]
        test_imgs = images[n_train:]

        for split_name, img_list in [("Train", train_imgs), ("Test", test_imgs)]:
            for img_path in img_list:
                label_path = img_path.with_suffix(".txt")
                
                if not label_path.exists():
                    print(f"⚠️ There are no labels for {img_path.name}, skip it.")
                    continue

                shutil.copy(img_path, output_root / split_name / "images" / img_path.name)
                shutil.copy(label_path, output_root / split_name / "labels" / label_path.name)

        print(f"Done process {folder.name}: {len(train_imgs)} train, {len(test_imgs)} test")

    data_yaml = {
        "train": "Train",
        "val": "Test",
        "nc": 1,  # change if we have more than 1 class
        "names": ["gun"]  # change with object we want to detect
    }

    with open(output_root / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\n✅ Done to split dataset and it saved at :", output_root)


if __name__ == "__main__":
    source_root = r"data_test_videos"
    output_root = r"data_test_video_yolo"
    split_yolo_dataset(source_root, output_root)
