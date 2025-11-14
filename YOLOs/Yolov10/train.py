# coding:utf-8
from ultralytics import YOLOv10
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10m.yaml"
# 数据集配置文件
data_yaml_path = r"/root/detection_dataset/dataset_v7/data.yaml"
# 预训练模型
pre_model_name = 'yolov10m.pt'


if __name__ == '__main__':
    # 加载预训练模型
    model = YOLOv10(pre_model_name)
    # 训练模型
    results = model.train(data=data_yaml_path,
                          epochs=20,
                          batch=32,
                          workers=1,
                          name='Seal_Detection_Train',
                          device='0',
                          optimizer='AdamW',
                          cos_lr=True,  # 使用余弦学习率调度
                          augment=True,  # 启用数据增强
                          single_cls=True,)
