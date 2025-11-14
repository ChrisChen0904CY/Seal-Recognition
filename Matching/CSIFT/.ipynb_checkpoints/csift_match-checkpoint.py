import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# 形态学闭运算
def apply_morphology(img, kernel_size_open=3, kernel_size_close=5, area_thresh=500):
    # Step 1: Canny
    edges = cv2.Canny(img, 50, 150)

    # Step 2: 闭运算（小核）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 3: 找轮廓，选最大圆形区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius > 30:  # 阈值避免误检
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)

    # Step 4: 用掩膜提取印章区域
    stamp_region = cv2.bitwise_and(closed, closed, mask=mask)
    return stamp_region

# 加载图像并提取 SIFT 特征（闭运算后）
def load_images_and_features(folder):
    images = {}
    features = {}
    sift = cv2.SIFT_create()
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                closed_img = apply_morphology(img)
                images[filename] = closed_img
                kp, des = sift.detectAndCompute(closed_img, None)
                features[filename] = (kp, des)
    return images, features

# 计算 SIFT 相似度
def compute_sift_similarity(des1, des2):
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [
        match[0] for match in matches
        if len(match) == 2 and match[0].distance < 0.75 * match[1].distance
    ]
    score = len(good_matches) / max(len(des1), len(des2))
    return score

# 匹配所有图像对
def match_all_pairs(features):
    image_names = list(features.keys())
    similarity_scores = []
    pbar = tqdm(total=(len(image_names)*(len(image_names)-1))//2, desc="Matching", ncols=100)
    for i in range(len(image_names)):
        for j in range(i+1, len(image_names)):
            name1 = image_names[i]
            name2 = image_names[j]
            des1 = features[name1][1]
            des2 = features[name2][1]
            score = compute_sift_similarity(des1, des2)
            similarity_scores.append((name1, name2, score))
            pbar.set_postfix_str(f"{name1} vs {name2}")
            pbar.update(1)
    return similarity_scores

# 绘制闭运算示例（前 6 张）
def plot_closing_examples(images, output_file="close.png"):
    sample_imgs = list(images.values())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, img in enumerate(sample_imgs):
        row, col = divmod(i, 3)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {i+1}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)

# 绘制 Top N 匹配结果
def plot_top_matches(similarity_scores, images, title, output_file, top_n=20, score_range=(0.7, 0.9)):
    max_score = max([s for _, _, s in similarity_scores])
    if max_score < score_range[0]:
        top_matches = sorted(similarity_scores, key=lambda x: x[2], reverse=True)[:top_n]
    else:
        filtered_matches = [
            (name1, name2, score)
            for name1, name2, score in similarity_scores
            if score_range[0] <= score <= score_range[1]
        ]
        top_matches = sorted(filtered_matches, key=lambda x: x[2], reverse=True)[:top_n]

    fig, axes = plt.subplots(5, 8, figsize=(20, 15))
    fig.suptitle(title, fontsize=18)

    for idx, (name1, name2, score) in enumerate(top_matches):
        img1 = images[name1]
        img2 = images[name2]
        row = idx // 4
        col = (idx % 4) * 2

        axes[row, col].imshow(img1, cmap='gray')
        axes[row, col].set_title(f"{name1}\nScore: {score:.2f}")
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(img2, cmap='gray')
        axes[row, col + 1].set_title(f"{name2}")
        axes[row, col + 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=200)

# 主函数
def main():
    fake_folder = '/root/CDataset/Fake'
    real_folder = '/root/CDataset/Real'

    # Fake
    fake_images, fake_features = load_images_and_features(fake_folder)
    plot_closing_examples(fake_images, "close.png")  # 保存闭运算示例
    fake_scores = match_all_pairs(fake_features)
    plot_top_matches(fake_scores, fake_images, "Top 20 Fake Matches", "sift_fake_result.png")

    # Real
    real_images, real_features = load_images_and_features(real_folder)
    real_scores = match_all_pairs(real_features)
    plot_top_matches(real_scores, real_images, "Top 20 Real Matches", "sift_real_result.png")

    # 保存分数
    with open("sift_fake_scores.json", "w") as f:
        json.dump([{ "img1": a, "img2": b, "score": s } for a, b, s in fake_scores], f, indent=2)
    with open("sift_real_scores.json", "w") as f:
        json.dump([{ "img1": a, "img2": b, "score": s } for a, b, s in real_scores], f, indent=2)

if __name__ == "__main__":
    main()
    