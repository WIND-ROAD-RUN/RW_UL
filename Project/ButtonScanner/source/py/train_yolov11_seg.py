from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt') 

    model.train(
        data='./train_yolov11_seg.yaml',  
        epochs=50,           
        batch=8,             
        imgsz=640,           
        device=0,         
        task='segment'       
    )

    metrics = model.val(task='segment') 

    model.export(format='onnx') 