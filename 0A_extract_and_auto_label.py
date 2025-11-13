import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
video_folder = 'test videos'                   # Folder contains video.
output_folder = 'data_test_videos/dataset'     # Folder output (images and txt).
model_path = r'models/real_models_1.pt'
target_classes = [0]                           # if want to detect specific class (like 0 = person), None for all classes.
frames_per_second = 1                          # how many frames you want to get from every seconds of video.

model = YOLO(model_path)
os.makedirs(output_folder, exist_ok=True)

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # counting how many frame that we will get from the video
    duration = total_frames / fps
    frames_to_extract = int(duration * frames_per_second)
    
    frame_interval = total_frames / frames_to_extract

    frame_num = 0

    video_output_dir = os.path.join(output_folder, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    for i in range(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_interval))
        ret, frame = cap.read()
        
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        frame_filename = f"{video_name}_{frame_num:05d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        label_filename = frame_filename.replace('.jpg', '.txt')
        label_path = os.path.join(video_output_dir, label_filename)
        with open(label_path, 'w') as f:
            for box in results.boxes:
                cls = int(box.cls)
                if target_classes is None or cls in target_classes:
                    x_center, y_center, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        frame_num += 1

    cap.release()
    print(f"Selesai: {video_name}")

video_exts = ['.mp4', '.avi', '.mov', '.mkv']
video_files = [f for f in os.listdir(video_folder) if any(f.lower().endswith(ext) for ext in video_exts)]

print(f"🔍 found {len(video_files)} video in folder '{video_folder}'")

for video_file in tqdm(video_files, desc="Total Progress", unit="video"):
    video_path = os.path.join(video_folder, video_file)
    process_video(video_path)

print("✅ all video already processed.")
