from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/NEU-DETTLFINet_mode300sz224bs323/weights/last.pt")  # build a new model from scratch

    # Use the model
    model.train(data="NEU-DET.yaml",
                epochs=300,
                batch=32,
                imgsz=224,
                patience=150,
                workers=8,
                name='NEU-DETTLFINet_mode300sz224bs32',
                resume=True,
                close_mosaic=15)
