from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/detect/新终改NEU-DETTLFINet_mode300sz416bs32/weights/best.pt')

    # Customize validation settings
    validation_results = model.val(data='NEU-DET.yaml',
                                imgsz=416,
                                batch=32,
                                conf=0.001,
                                iou=0.5,
                                device='0')
