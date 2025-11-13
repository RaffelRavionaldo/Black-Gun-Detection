import os
import shutil
from tqdm import tqdm

# === CONFIG ===
input_folder = r"synthetic_dataset_KAC_PDW_Blackgun"
output_folder = r"clean_synthetic_dataset_KAC_PDW_Blackgun"

# === Collect all files to be copied (without '_aug') =
all_files = []
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if "_aug" not in file:
            all_files.append(os.path.join(subdir, file))

print(f"Found {len(all_files)} non-augmented files to copy.\n")

# === Copy files with progress bar ===
for src_path in tqdm(all_files, desc="Copying clean dataset"):
    dst_path = src_path.replace(input_folder, output_folder)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

print("\n✅ Dataset cleaned successfully!")
print(f"Output saved in: {output_folder}")