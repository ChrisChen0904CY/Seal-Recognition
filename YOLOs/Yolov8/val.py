from ultralytics import YOLO
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 使用YOLOv11进行训练
def train_yolov8():
    # 迁移训练 加载权重
    model = YOLO("best.pt")

    # 训练
    model.val(
        data=r"/root/detection_dataset/dataset_v7/data_test.yaml",
        epochs=20,  # 设置训练轮数
        imgsz=640,  # 设置图片大小
        batch=32,  # 批次大小
        device='0',
    )


# 启动训练
if __name__ == '__main__':
    train_yolov8()
