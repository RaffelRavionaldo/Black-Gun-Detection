from ultralytics import YOLO
import torch

def main():
    # Clear the cache
    torch.cuda.empty_cache()

    # model = YOLO("yolov8n.pt")
    model = YOLO('runs/detect/fine_tuned_synthentic_model_with_real_data/weights/best.pt')
    data_yaml = r'data_test_video_yolo\data.yaml'

    # start training
    model.train(
        data=data_yaml,
        epochs=51,
        imgsz=640,
        batch=-1,
        save_period=50,
        degrees=30,
        shear=5,
        flipud=0.1,
        fliplr=0.7,
        mosaic=0.25,
        erasing=0.0,
        hsv_h=0.002,
        hsv_s=0.1,
        hsv_v=0.1
    )


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
