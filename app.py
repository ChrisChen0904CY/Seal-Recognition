from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F
from PIL import Image
import base64
from io import BytesIO
import glob
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from datetime import datetime

# 导入自定义模块
from Classification.model.model import StampClassifier
from Matching.CSIFT.csift_match import apply_morphology, compute_sift_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEALS_FOLDER'] = 'seals_library'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4G

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEALS_FOLDER'], exist_ok=True)

# 全局变量
yolo_model = None
classifier_model = None
seals_data = None
device = None
SEALS_JSON_PATH = 'seals.json'


# 初始化模型和印章数据库
def init_models():
	global yolo_model, classifier_model, seals_data, device

	# 设置设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"使用设备: {device}")

	# 加载YOLOv11模型
	try:
		from ultralytics import YOLO
		yolo_model = YOLO('detection_results/yolov11_exp/runs/detect/train2/weights/best.pt')
		if torch.cuda.is_available():
			yolo_model.to('cuda')
		print("YOLOv11模型加载成功")
	except Exception as e:
		print(f"YOLOv11模型加载失败: {e}")
		yolo_model = None

	# 加载分类模型
	try:
		classifier_model = StampClassifier()
		classifier_model.load_state_dict(torch.load('Classification/weights/best.pt', map_location=device))
		classifier_model.to(device)
		classifier_model.eval()
		print("分类模型加载成功")
	except Exception as e:
		print(f"分类模型加载失败: {e}")
		classifier_model = None

	# 加载或创建印章数据库
	seals_data = load_or_create_seals_database()


def load_or_create_seals_database():
	"""加载或创建印章数据库"""
	if os.path.exists(SEALS_JSON_PATH):
		try:
			with open(SEALS_JSON_PATH, 'r', encoding='utf-8') as f:
				data = json.load(f)
			print(f"印章数据库加载成功，共 {len(data)} 个组")
			return data
		except Exception as e:
			print(f"印章数据库加载失败，创建新数据库: {e}")
			return create_empty_seals_database()
	else:
		print("未找到印章数据库，创建新数据库")
		return create_empty_seals_database()


def create_empty_seals_database():
	"""创建空的印章数据库结构"""
	return {
		"metadata": {
			"created_at": datetime.now().isoformat(),
			"version": "1.0",
			"total_groups": 0,
			"total_seals": 0
		},
		"groups": {}
	}


def save_seals_database():
	"""保存印章数据库"""
	try:
		# 更新元数据
		seals_data["metadata"]["updated_at"] = datetime.now().isoformat()
		seals_data["metadata"]["total_groups"] = len(seals_data["groups"])
		seals_data["metadata"]["total_seals"] = sum(
			len(group["members"]) for group in seals_data["groups"].values()
		)

		with open(SEALS_JSON_PATH, 'w', encoding='utf-8') as f:
			json.dump(seals_data, f, ensure_ascii=False, indent=2)
		print("印章数据库保存成功")
		return True
	except Exception as e:
		print(f"印章数据库保存失败: {e}")
		return False


def save_seal_image(stamp_image, seal_id):
	"""保存印章图像到库中"""
	seal_filename = f"{seal_id}.png"
	seal_path = os.path.join(app.config['SEALS_FOLDER'], seal_filename)

	try:
		stamp_image.save(seal_path, 'PNG')
		return seal_path
	except Exception as e:
		print(f"保存印章图像失败: {e}")
		return None


def find_best_matching_group(stamp_features, threshold=0.8):
	"""找到最佳匹配的印章组"""
	if not seals_data["groups"]:
		return None, 0.0

	best_group = None
	best_similarity = 0.0

	sift = cv2.SIFT_create()
	kp_stamp, des_stamp = stamp_features

	for group_name, group_data in seals_data["groups"].items():
		group_similarities = []

		for member in group_data["members"]:
			seal_path = member.get("path", "")
			if os.path.exists(seal_path):
				try:
					seal_img = cv2.imread(seal_path, cv2.IMREAD_GRAYSCALE)
					if seal_img is None:
						continue

					processed_seal = apply_morphology(seal_img)
					kp_seal, des_seal = sift.detectAndCompute(processed_seal, None)

					if des_seal is not None and des_stamp is not None:
						similarity = compute_sift_similarity(des_stamp, des_seal)
						group_similarities.append(similarity)
				except Exception as e:
					print(f"比较印章时出错: {e}")
					continue

		if group_similarities:
			avg_similarity = max(group_similarities)  # 使用最大相似度
			if avg_similarity > best_similarity:
				best_similarity = avg_similarity
				best_group = group_name

	return best_group, best_similarity


def register_new_seal(stamp_image, authenticity_prob):
	"""注册新印章到数据库"""
	# 提取特征
	stamp_cv = pil_to_cv2(stamp_image)
	stamp_gray = cv2.cvtColor(stamp_cv, cv2.COLOR_BGR2GRAY)
	processed_stamp = apply_morphology(stamp_gray)

	sift = cv2.SIFT_create()
	stamp_features = sift.detectAndCompute(processed_stamp, None)

	# 寻找最佳匹配组
	best_group, best_similarity = find_best_matching_group(stamp_features)

	# 生成唯一ID
	seal_id = f"seal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

	# 保存印章图像
	seal_path = save_seal_image(stamp_image, seal_id)
	if not seal_path:
		return None, "保存印章图像失败"

	# 创建印章记录
	seal_record = {
		"id": seal_id,
		"path": seal_path,
		"authenticity": authenticity_prob,
		"added_at": datetime.now().isoformat(),
		"features": {
			"keypoints_count": len(stamp_features[0]) if stamp_features[0] else 0
		}
	}

	# 决定是否创建新组或添加到现有组
	if best_group and best_similarity >= 0.9:
		# 添加到现有组
		seals_data["groups"][best_group]["members"].append(seal_record)
		seals_data["groups"][best_group]["updated_at"] = datetime.now().isoformat()
		group_name = best_group
		action = "added_to_existing"
	else:
		# 创建新组
		group_name = f"group_{len(seals_data['groups']) + 1:04d}"
		seals_data["groups"][group_name] = {
			"created_at": datetime.now().isoformat(),
			"updated_at": datetime.now().isoformat(),
			"members": [seal_record]
		}
		action = "created_new"

	# 保存数据库
	if save_seals_database():
		return group_name, action
	else:
		return None, "数据库保存失败"


# 图像处理工具函数（保持不变）
def pil_to_cv2(pil_image):
	return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
	return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def image_to_base64(image):
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode()


# YOLO检测函数（保持不变）
def detect_stamps(image):
	if yolo_model is None:
		return []

	try:
		results = yolo_model(image)
		detections = []

		for result in results:
			boxes = result.boxes
			if boxes is not None:
				for box in boxes:
					x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
					conf = box.conf[0].cpu().numpy()
					detections.append({
						'bbox': [float(x1), float(y1), float(x2), float(y2)],
						'confidence': float(conf)
					})

		return detections
	except Exception as e:
		print(f"YOLO检测失败: {e}")
		return []


# 真伪鉴别函数（保持不变）
def authenticate_stamp(stamp_image):
	if classifier_model is None:
		return 0.5

	stamp_image = stamp_image.resize((224, 224))
	image_array = np.array(stamp_image)

	if len(image_array.shape) == 2:
		image_array = np.stack([image_array] * 3, axis=-1)
	elif image_array.shape[2] == 4:
		image_array = image_array[:, :, :3]

	image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
	image_tensor = image_tensor.unsqueeze(0).to(device)

	with torch.no_grad():
		try:
			output = classifier_model(image_tensor)
			real_prob = output[0][1].item()
		except Exception as e:
			print(f"分类推理失败: {e}")
			real_prob = 0.5

	return real_prob


# 配准函数（更新为使用数据库）
def register_stamp(stamp_image, top_k=5):
	if not seals_data["groups"]:
		return []

	try:
		# 提取SIFT特征
		stamp_cv = pil_to_cv2(stamp_image)
		stamp_gray = cv2.cvtColor(stamp_cv, cv2.COLOR_BGR2GRAY)
		processed_stamp = apply_morphology(stamp_gray)

		sift = cv2.SIFT_create()
		kp_stamp, des_stamp = sift.detectAndCompute(processed_stamp, None)

		if des_stamp is None:
			return []

		similarities = []

		# 计算与所有已知印章的相似度
		for group_name, group_data in seals_data["groups"].items():
			group_similarities = []
			best_member = None

			for member in group_data.get("members", []):
				seal_path = member.get("path", "")
				if os.path.exists(seal_path):
					try:
						seal_img = cv2.imread(seal_path, cv2.IMREAD_GRAYSCALE)
						if seal_img is None:
							continue

						processed_seal = apply_morphology(seal_img)
						kp_seal, des_seal = sift.detectAndCompute(processed_seal, None)

						if des_seal is not None:
							similarity = compute_sift_similarity(des_stamp, des_seal)
							group_similarities.append(similarity)

							# 记录最佳匹配的成员
							if not best_member or similarity > best_member["similarity"]:
								best_member = {
									"member": member,
									"similarity": similarity,
									"path": seal_path
								}
					except Exception as e:
						print(f"处理印章 {seal_path} 时出错: {e}")
						continue

			if best_member and group_similarities:
				max_similarity = max(group_similarities)
				similarities.append({
					'group': group_name,
					'seal_path': best_member["path"],
					'similarity': max_similarity,
					'seal_info': best_member["member"],
					'group_size': len(group_data.get("members", []))
				})

		# 按相似度排序并返回前top_k个
		similarities.sort(key=lambda x: x['similarity'], reverse=True)
		return similarities[:top_k]

	except Exception as e:
		print(f"配准处理失败: {e}")
		return []


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
	if 'image' not in request.files:
		return jsonify({'error': '没有上传文件'}), 400

	file = request.files['image']
	if file.filename == '':
		return jsonify({'error': '没有选择文件'}), 400

	# 保存上传的图像
	filename = secure_filename(file.filename)
	filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(filepath)

	# 打开图像
	try:
		original_image = Image.open(filepath)
		original_cv = pil_to_cv2(original_image)
	except Exception as e:
		return jsonify({'error': f'图像打开失败: {e}'}), 400

	# 步骤1: 印章检测
	detections = detect_stamps(original_cv)

	results = []

	# 在原图上绘制检测框
	result_image = original_cv.copy()

	for i, detection in enumerate(detections):
		x1, y1, x2, y2 = detection['bbox']
		confidence = detection['confidence']

		# 提取印章区域
		stamp_region = original_cv[int(y1):int(y2), int(x1):int(x2)]
		stamp_pil = cv2_to_pil(stamp_region)

		# 步骤2: 真伪鉴别
		authenticity_prob = authenticate_stamp(stamp_pil)

		# 步骤3: 配准
		top_matches = register_stamp(stamp_pil)

		# 步骤4: 注册到印章数据库
		group_name, action = register_new_seal(stamp_pil, authenticity_prob)

		# 绘制检测框和标签
		cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
		label = f'Stamp {i + 1}: {confidence:.2f}'
		cv2.putText(result_image, label, (int(x1), int(y1) - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# 将印章区域转为base64
		stamp_base64 = image_to_base64(stamp_pil)

		# 处理匹配结果
		matches_data = []
		for match in top_matches:
			try:
				match_img = Image.open(match['seal_path'])
				match_base64 = image_to_base64(match_img.resize((100, 100)))
				matches_data.append({
					'image': match_base64,
					'similarity': match['similarity'],
					'group': match['group'],
					'group_size': match.get('group_size', 1)
				})
			except Exception as e:
				print(f"加载匹配图像失败: {e}")
				continue

		results.append({
			'stamp_id': i + 1,
			'bbox': detection['bbox'],
			'confidence': confidence,
			'stamp_image': stamp_base64,
			'authenticity': authenticity_prob,
			'matches': matches_data,
			'registration': {
				'group': group_name,
				'action': action
			}
		})

	# 将结果图像转为base64
	result_pil = cv2_to_pil(result_image)
	result_base64 = image_to_base64(result_pil)

	# 清理临时文件
	try:
		os.remove(filepath)
	except:
		pass

	return jsonify({
		'original_image': image_to_base64(original_image),
		'result_image': result_base64,
		'detections': results,
		'database_info': {
			'total_groups': seals_data["metadata"]["total_groups"],
			'total_seals': seals_data["metadata"]["total_seals"]
		}
	})


# 文件夹处理路由（简化版，只检测不注册）
@app.route('/process_folder', methods=['POST'])
def process_folder():
	if 'folder' not in request.files:
		return jsonify({'error': '没有上传文件夹'}), 400

	files = request.files.getlist('folder')
	if not files or len(files) == 0:
		return jsonify({'error': '文件夹为空或没有选择文件'}), 400

	# 限制一次处理的文件数量，避免内存溢出
	MAX_FILES_PER_BATCH = 20
	if len(files) > MAX_FILES_PER_BATCH:
		return jsonify({
			'error': f'文件数量过多，一次最多处理 {MAX_FILES_PER_BATCH} 个文件',
			'uploaded_count': len(files),
			'max_allowed': MAX_FILES_PER_BATCH
		}), 400

	all_results = {}
	processed_count = 0

	for file in files:
		if file.filename == '' or not file.filename:
			continue

		filename = secure_filename(file.filename)
		if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
			continue

		print(f"处理文件: {filename}")

		# 检查文件大小
		file.seek(0, 2)  # 移动到文件末尾
		file_size = file.tell()
		file.seek(0)  # 重置文件指针

		MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
		if file_size > MAX_FILE_SIZE:
			all_results[filename] = {
				'error': f'文件过大 ({file_size / (1024 * 1024):.1f}MB)，跳过处理',
				'status': 'skipped'
			}
			continue

		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		try:
			file.save(filepath)
		except Exception as e:
			all_results[filename] = {'error': f'文件保存失败: {str(e)}', 'status': 'error'}
			continue

		try:
			original_image = Image.open(filepath)
			if original_image.mode != 'RGB':
				original_image = original_image.convert('RGB')

			original_cv = pil_to_cv2(original_image)

			detections = detect_stamps(original_cv)
			file_results = []

			for i, detection in enumerate(detections):
				x1, y1, x2, y2 = detection['bbox']
				h, w = original_cv.shape[:2]
				x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))

				if x2 <= x1 or y2 <= y1:
					continue

				stamp_region = original_cv[y1:y2, x1:x2]
				if stamp_region.size == 0:
					continue

				stamp_pil = cv2_to_pil(stamp_region)
				authenticity_prob = authenticate_stamp(stamp_pil)

				# 在文件夹处理中不进行配准和注册，只做检测和鉴别
				file_results.append({
					'stamp_id': i + 1,
					'bbox': [x1, y1, x2, y2],
					'confidence': detection['confidence'],
					'authenticity': authenticity_prob
				})

			all_results[filename] = {
				'detection_count': len(detections),
				'stamps': file_results,
				'status': 'success'
			}
			processed_count += 1

		except Exception as e:
			all_results[filename] = {
				'error': f'处理失败: {str(e)}',
				'status': 'error'
			}

		# 立即清理临时文件，释放内存
		finally:
			try:
				if os.path.exists(filepath):
					os.remove(filepath)
			except Exception as e:
				print(f"清理文件失败: {e}")

	return jsonify({
		'results': all_results,
		'summary': {
			'total_files': len(files),
			'processed_files': processed_count,
			'successful_files': len([r for r in all_results.values() if r.get('status') == 'success']),
			'skipped_files': len([r for r in all_results.values() if r.get('status') == 'skipped']),
			'error_files': len([r for r in all_results.values() if r.get('status') == 'error'])
		}
	})


@app.route('/start_batch_processing', methods=['POST'])
def start_batch_processing():
	"""开始批量处理 - 接收文件列表但不立即处理"""
	if 'file_list' not in request.json:
		return jsonify({'error': '没有文件列表'}), 400

	file_list = request.json['file_list']
	if not file_list:
		return jsonify({'error': '文件列表为空'}), 400

	# 这里可以保存文件列表到数据库或session中，然后分批处理
	# 简化版本：直接返回文件列表确认
	return jsonify({
		'message': '批量处理任务已接收',
		'file_count': len(file_list),
		'task_id': f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	})


@app.route('/upload_single_for_batch', methods=['POST'])
def upload_single_for_batch():
	"""为批量处理单独上传一个文件"""
	print("=== 开始处理单个文件 ===")
	print(f"请求文件: {request.files}")

	if 'file' not in request.files:
		print("错误: 没有找到 'file' 字段")
		return jsonify({'error': '没有上传文件'}), 400

	file = request.files['file']
	print(f"接收到文件: {file.filename}, 大小: {file.content_length if file.content_length else '未知'}")

	if file.filename == '':
		print("错误: 文件名为空")
		return jsonify({'error': '没有选择文件'}), 400

	filename = secure_filename(file.filename)
	print(f"安全文件名: {filename}")

	filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

	try:
		# 保存文件
		file.save(filepath)
		print(f"文件保存成功: {filepath}")

		# 检查文件是否存在
		if not os.path.exists(filepath):
			raise Exception("文件保存后不存在")

		original_image = Image.open(filepath)
		if original_image.mode != 'RGB':
			original_image = original_image.convert('RGB')

		original_cv = pil_to_cv2(original_image)

		detections = detect_stamps(original_cv)
		file_results = []

		for i, detection in enumerate(detections):
			x1, y1, x2, y2 = detection['bbox']
			stamp_region = original_cv[int(y1):int(y2), int(x1):int(x2)]
			stamp_pil = cv2_to_pil(stamp_region)

			authenticity_prob = authenticate_stamp(stamp_pil)

			# 真伪判断
			authenticity_status = "伪造" if authenticity_prob < 0.5 else "真实"

			file_results.append({
				'stamp_index': i + 1,  # 文件内的印章索引
				'bbox': [int(x1), int(y1), int(x2), int(y2)],
				'confidence': detection['confidence'],
				'authenticity_prob': authenticity_prob,
				'authenticity_status': authenticity_status
			})

		# 清理临时文件
		if os.path.exists(filepath):
			os.remove(filepath)

		print(f"文件处理成功: {filename}, 检测到 {len(detections)} 个印章")

		return jsonify({
			'filename': filename,
			'detection_count': len(detections),
			'stamps': file_results,
			'status': 'success'
		})

	except Exception as e:
		# 清理临时文件
		if os.path.exists(filepath):
			try:
				os.remove(filepath)
			except:
				pass
		print(f"文件处理失败: {filename}, 错误: {str(e)}")
		return jsonify({
			'filename': filename,
			'error': str(e),
			'status': 'error'
		}), 500


@app.route('/database_info')
def get_database_info():
	"""获取数据库信息"""
	return jsonify({
		'total_groups': seals_data["metadata"]["total_groups"],
		'total_seals': seals_data["metadata"]["total_seals"],
		'last_updated': seals_data["metadata"].get("updated_at", seals_data["metadata"]["created_at"])
	})


if __name__ == '__main__':
	print("正在初始化模型...")
	init_models()
	print("启动Flask应用...")
	app.run(debug=True, host='0.0.0.0', port=5000)
