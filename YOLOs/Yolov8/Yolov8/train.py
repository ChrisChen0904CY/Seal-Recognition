from ultralytics import YOLO
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 使用YOLOv11进行训练
def train_yolov8():
    # 迁移训练 加载权重
    model = YOLO("yolov8m.pt")

    # 训练
    model.train(
        data='P:/Graduation_Design/Dataset/JHU_CROWD/jhu_crowd_v2.0/jhu_crowd_v2.0/data.yaml',
        epochs=2,  # 设置训练轮数
        imgsz=640,  # 设置图片大小
        batch=2,  # 批次大小
        device='0',
        workers=1,
        optimizer='AdamW',
        cos_lr=True,  # 使用余弦学习率调度
        augment=True,  # 启用数据增强
        single_cls=True,
    )


# 启动训练
if __name__ == '__main__':
    train_yolov8()
