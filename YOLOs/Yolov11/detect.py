from ultralytics import YOLO
import os
import cv2

# 加载YOLOv11模型
model = YOLO("./params/best.pt")

# 图像目录路径
image_dir = "P:/FintechComp/Detect_Test"
# 检出结果导出目录
save_dir = "./runs/exp"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 遍历目录中的所有图像文件
for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):  # 处理图片格式
        img_path = os.path.join(image_dir, img_file)

        # 读取原始图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        img_height, img_width = img.shape[:2]  # 获取图像尺寸

        # 使用YOLO模型进行推理
        results = model(img_path, classes=[0])[0]  # 获取第一个推理结果

        # 为当前图片创建对应的结果文件夹
        img_name = img_file.rsplit(".", 1)[0]  # 获取不带后缀的文件名
        img_save_dir = os.path.join(save_dir, img_name)
        os.makedirs(img_save_dir, exist_ok=True)

        # 保存推理后的整张图片
        results.save(os.path.join(img_save_dir, img_file))

        # 文本文件路径
        txt_file_path = os.path.join(img_save_dir, "boxes.txt")

        # 解析检测框信息并按置信度排序（降序）
        detections = []
        for box in results.boxes.data:
            x_min, y_min, x_max, y_max, conf = box[:5]
            detections.append((int(x_min), int(y_min), int(x_max), int(y_max), float(conf)))

        detections.sort(key=lambda x: x[4], reverse=True)  # 按置信度排序

        # 写入检测框数据
        with open(txt_file_path, "w") as f:
            for i, (x_min, y_min, x_max, y_max, conf) in enumerate(detections, start=1):
                f.write(f"{x_min} {y_min} {x_max} {y_max} {conf:.4f}\n")

                # 裁剪检测出的目标区域
                cropped_img = img[y_min:y_max, x_min:x_max]

                # 保存裁剪后的图片
                crop_filename = os.path.join(img_save_dir, f"{i}.jpg")
                cv2.imwrite(crop_filename, cropped_img)
