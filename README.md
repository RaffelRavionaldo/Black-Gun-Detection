# Black Gun Detection with yolov8

This library was created with Python 3.10.

## Install the library needed

```
pip install ultralytics==8.3.162 opencv-python==4.11.0.86 tqdm==4.66.5

# if you only want to use CPU
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# if you want to use a GPU
## check cuda
nvcc --version
```

The output will look like : 

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_
```

So in my case, I installed Torch with CUDA 12.4 :

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## Code explanation

### 0A_extract_and_auto_label

It's for getting images from video that you have and doing auto-labeling them with yolo models, but you need to check if the result is already perfect or needs some fix (you can use LabelImg, Roboflow, cvat, etc, for checking it).

Before running it, you need to change the configuration (this is in the code) :
```
video_folder = 'test videos'                   # Folder contains video.
output_folder = 'data_test_videos/dataset'     # Folder output (images and txt).
model_path = r'models/real_models_1.pt'
target_classes = [0]                           # if want to detect specific class (like 0 = person), None for all classes.
frames_per_second = 1                          # how many frames you want to get from every second of video.
```

and then simply run `python 0A_extract_and_auto_label.py`

### 0_clean_data.py

In my case, the data has augmentation that I don't need, and luckily, the augmented data contains _aug for that. so just copy and paste images and labels that no contains that. for run it just type `python 0_clean_data.py`

### 1_split_data.py

My dataset folder structure looks like below : 

```
--- Folder_A
------- dataset
------------ images
---------------------- images_1.jpg
----------- labels
---------------------- images_1.txt
--- Folder_B
--- Folder_C
```

So this code combines all datasets we have, and splits them into 2 parts, train and validation, then creates the data.yaml so we don't need to create it manually.

Before running the code, you need to change the config : 

```
# the name of your dataset folder
input_folders = [
    r"real_dataset_Nerf_Bluegun",
    r"synthetic_dataset_Nerf_Bluegun",
    r"coco"
]
# where you want to save it
output_folder = r"dataset_Nerf_Bluegun_used"
# it's to split data, in this case 80% will be training data and the rest for validation
train_part = 0.8
test_part = 1 - train_part
```

And on line 49, change the nc and names for data.yaml if necessary.

### 1_split_data_2.py

The function is the same as the previous code, but because we have a different folder format (it's the output from  0A_extract_and_auto_label code), so we need to create new code.

```
--- Folder_D
----- dataset
----------- video_a
---------------------- images
------------------------------ images.jpg
---------------------- labels
----------------------------- labels.txt
------------ video_b
------------ video_c
```

### 2_train_yolov8n.py

It's just for training the yolo models, but you will need the parameters according to what you need. You can understand it by reading this documentation : https://docs.ultralytics.com/guides/yolo-data-augmentation/#color-space-augmentations

for train it from "scratch". You can change the config : 

from 
```
model = YOLO('runs/detect/fine_tuned_synthentic_model_with_real_data/weights/best.pt') # is the path of previous training or others models that you already train
data_yaml = r'data_test_video_yolo\data.yaml' # path to your data.yaml, change it with yours
```

to 
```
model = YOLO('yolov8n.pt')
data_yaml = r'data_test_video_yolo\data.yaml' # path to your data.yaml, change it with yours

```

### 3_inference

This is the code for using our models to detect the video we want to test. In the main, you need to change the code : 

```
if __name__ == "__main__":
    model_path = r"models/test_models.pt" # it can be a folder that contains all yolo models, or only specific models
    video_path = r"test videos/"          # same, it can be a folder or specific video
    main(model_path, video_path)
```

The output will be in the "output" folders with this format : 

```
outout
----- model_A
----------- video_A.mp4
----------- video_B.mp4
----- model_B
----------- video_A.mp4
----------- video_B.mp4
```
