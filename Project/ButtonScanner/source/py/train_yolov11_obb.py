from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO('yolo11m.pt')  # 使用预训练模型

    # 开始训练
    model.train(
        data='./train_yolov11_obb.yaml',  # 数据集配置文件
        epochs=100,           # 训练轮数
        batch=16,             # 批量大小
        imgsz=640,            # 输入图像大小
        device=0,            # 使用 GPU (0 表示第一个 GPU)
        verbose=True
    )

    # 验证模型
    metrics = model.val()

    # 导出模型
    model.export(format='onnx')  # 导出为 ONNX 格式