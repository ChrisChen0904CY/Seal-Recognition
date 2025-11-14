from ultralytics import YOLOv10
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 使用YOLOv10进行训练
def val_yolov10():
    # 迁移训练 加载权重
    model = YOLOv10("last.pt")

    # 训练
    model.val(
        data=r"/root/detection_dataset/dataset_v7/data_test.yaml",
        imgsz=640,  # 设置图片大小
        batch=32,  # 批次大小
        device='0',
        workers=1,
        optimizer='AdamW',
        cos_lr=True,  # 使用余弦学习率调度
        augment=True,  # 启用数据增强
        single_cls=True,
    )


# 启动训练
if __name__ == '__main__':
    val_yolov10()
