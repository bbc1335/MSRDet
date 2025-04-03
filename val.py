from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('best.pt')

    # Customize validation settings
    validation_results = model.val(data='NEU-DET.yaml',
                                imgsz=224,
                                batch=32,
                                conf=0.001,
                                iou=0.5,
                                device='0')
