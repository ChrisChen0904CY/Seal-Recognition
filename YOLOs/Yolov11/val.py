from ultralytics import YOLO
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 使用YOLOv11进行训练
def val_yolov11():
    # 迁移训练 加载权重
    model = YOLO("./params/head250m-best.pt")

    # 训练
    model.train(
        data=r'S:\ClassHeadDetect\Class_Head_Dataset\CHD\data.yaml',
        imgsz=640,  # 设置图片大小
        batch=4,  # 批次大小
        device='0',
        workers=1,
        optimizer='AdamW',
        cos_lr=True,  # 使用余弦学习率调度
        augment=True,  # 启用数据增强
        single_cls=True,
    )


# 启动训练
if __name__ == '__main__':
    val_yolov11()
