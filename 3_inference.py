import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def get_model_list(model_path):
    model_path = Path(model_path).resolve()
    if model_path.is_dir():
        pt_files = list(model_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt models at {model_path}")
        return pt_files
    elif model_path.is_file():
        return [model_path]
    else:
        raise FileNotFoundError(f"Path model not valid: {model_path}")

def get_video_list(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov"))
    elif input_path.is_file():
        return [input_path]
    else:
        raise FileNotFoundError(f"not found any video at this path : {input_path}")

def process_video(video_path, model, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Fail to open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done: {output_path}")

def main(model_path, input_path, output_root="output"):
    model_files = get_model_list(model_path)
    videos = get_video_list(input_path)

    for model_file in model_files:
        model_name = Path(model_file).stem
        model = YOLO(model_file)
        print(f"\n=== Process with model : {model_name} ===")

        for video in videos:
            video_name = video.stem
            output_dir = Path(output_root) / model_name
            output_file = output_dir / f"{video_name}_output.mp4"
            process_video(video, model, output_file)

if __name__ == "__main__":
    model_path = r"models/test_models.pt"
    video_path = r"test videos/"
    main(model_path, video_path)