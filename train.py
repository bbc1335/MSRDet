from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("TLFINet_mod.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="GC-DET.yaml",
                epochs=300,
                batch=16,
                imgsz=512,
                patience=150,
                workers=8,
                name='新终改GC-DETTLFINet_mode300sz512bs16',
                save_period=1,
                resume=False,
                close_mosaic=0)
