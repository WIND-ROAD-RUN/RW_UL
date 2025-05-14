from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')  

    model.train(
        data='./train_yolov11_obb_Color.yaml', 
        epochs=100,           
        batch=16,             
        imgsz=640,           
        device=0,           
        verbose=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        lr0=0.01,
        lrf=0.017,
        scale=0.0,
        translate=0.3,
        mosaic=0.0,
        mixup=0.0
    )

    metrics = model.val()

    model.export(format='onnx')