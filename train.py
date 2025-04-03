from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("TLFINet.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="NEU-DET.yaml",
                epochs=300,
                batch=32,
                imgsz=224,
                patience=100,
                workers=8,
                close_mosaic=15)
