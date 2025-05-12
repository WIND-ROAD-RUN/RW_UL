from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11m.pt')  

    model.train(
        data='./train_yolov11_obb.yaml', 
        epochs=100,           
        batch=16,             
        imgsz=640,           
        device=0,           
        verbose=True
    )

    metrics = model.val()

    model.export(format='onnx')