from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练的 YOLOv11 分割模型
    model = YOLO('yolo11n-seg.pt')  # 使用预训练的语义分割模型

    # 开始训练
    model.train(
        data='C:/workPlace/project/trainModel/seg/dataset1.yaml',  # 数据集配置文件
        epochs=50,            # 训练轮数
        batch=8,              # 批量大小
        imgsz=640,            # 输入图像大小
        device=0,             # 使用 GPU (0 表示第一个 GPU)
        task='segment'        # 指定任务为语义分割
    )

    # 验证模型
    metrics = model.val(task='segment')  # 验证语义分割模型

    # 导出模型
    model.export(format='onnx')  # 导出为 ONNX 格式