import os
import shutil
import random


def gen_v5_dataset(source_dirs_, target_dir_, split_p=(0.8, 0.1, 0.1)):
	# 创建目标文件夹
	os.makedirs(target_dir_, exist_ok=True)
	os.makedirs(target_dir_+'/images', exist_ok=True)

	# 存储所有图像的绝对路径
	image_paths = []

	# 遍历源文件夹中的所有图像文件
	for source_dir in source_dirs_:
		for root, _, files in os.walk(source_dir):
			for file in files:
				if file.lower().endswith(('.png', '.jpg')):
					src_path = os.path.join(root, file)
					dst_path = os.path.join(target_dir_ + "/images", file)
					shutil.copy2(src_path, dst_path)
					image_paths.append(os.path.abspath(dst_path))

	# 打乱图像路径列表
	random.shuffle(image_paths)

	# 按指定的比例划分数据集
	total = len(image_paths)
	train_end = int(total * split_p[0])
	val_end = int(total * (split_p[0] + split_p[1]))

	train_paths = image_paths[:train_end]
	val_paths = image_paths[train_end:val_end]
	test_paths = image_paths[val_end:]

	# 将路径写入对应的txt文件（追加模式）
	with open(f'{target_dir_}/train.txt', 'a') as f:
		for path in train_paths:
			f.write(path.replace("\\", "/") + '\n')

	with open(f'{target_dir_}/val.txt', 'a') as f:
		for path in val_paths:
			f.write(path.replace("\\", "/") + '\n')

	with open(f'{target_dir_}/test.txt', 'a') as f:
		for path in test_paths:
			f.write(path.replace("\\", "/") + '\n')

	# 获取当前脚本所在位置的绝对路径
	abs_path = os.path.abspath('.')
	yaml_lines = [
		f"train: {os.path.join(abs_path, f'{target_dir_}/train.txt')}".replace('\\', '/'),
		f"val: {os.path.join(abs_path, f'{target_dir_}/val.txt')}".replace('\\', '/'),
		f"test: {os.path.join(abs_path, f'{target_dir_}/test.txt')}".replace('\\', '/'),
		"nc: 1",
		"names: ['Seal']"
	]

	with open(os.path.join('./dataset_v5', 'data.yaml'), 'w', encoding='utf-8') as f:
		f.write('\n'.join(yaml_lines))

	print("✅ 数据集处理完成，配置文件已生成。")


def gen_v7_dataset(source_dir_, target_dir_):
	splits = ['train', 'val', 'test']
	for split in splits:
		# 创建目标文件夹结构
		images_target_dir = os.path.join(target_dir_, split, 'images')
		labels_target_dir = os.path.join(target_dir_, split, 'labels')
		os.makedirs(images_target_dir, exist_ok=True)
		os.makedirs(labels_target_dir, exist_ok=True)

		# 读取对应的 txt 文件
		txt_file = os.path.join(source_dir_, f'{split}.txt')
		with open(txt_file, 'r') as f:
			image_paths = [line.strip() for line in f.readlines()]

		for img_path in image_paths:
			# 复制图像文件
			img_name = os.path.basename(img_path)
			shutil.copy(img_path, os.path.join(images_target_dir, img_name))

			# 构造标签路径并复制标签文件
			label_name = os.path.splitext(img_name)[0] + '.txt'
			label_path = os.path.join(source_dir_, 'labels', label_name)
			if os.path.exists(label_path):
				shutil.copy(label_path, os.path.join(labels_target_dir, label_name))
			else:
				print(f"⚠️ 标签文件不存在: {label_path}")

	print("✅ 数据集处理完成，配置文件已生成。")


if __name__ == "__main__":
    # 设置随机种子
	random.seed(1145114)

	# 设置源文件夹路径
	source_dirs = ['Seals/True/Detection_Data', 'Seals/Fake/Detection_Data']
	target_dir = 'dataset_v5'

	# 生成 v5 风格数据集以及相关配置文件
	# gen_v5_dataset(source_dirs, target_dir)
	# 生成 v7 风格数据集以及相关配置文件
	gen_v7_dataset(target_dir, 'dataset_v7')