import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm


def load_images_and_features(folder):
    images = {}
    features = {}
    sift = cv2.SIFT_create()
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[filename] = img
                kp, des = sift.detectAndCompute(img, None)
                features[filename] = (kp, des)
    return images, features


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


def plot_top_matches(similarity_scores, images, title, output_file, top_n=20, score_range=(0.7, 0.9)):
    # 判断是否有匹配对满足阈值范围
    max_score = max([s for _, _, s in similarity_scores])
    if max_score < score_range[0]:
        # 如果最大值都小于阈值下限，直接取前 top_n 对
        top_matches = sorted(similarity_scores, key=lambda x: x[2], reverse=True)[:top_n]
    else:
        # 否则筛选在指定范围内的匹配对
        filtered_matches = [
            (name1, name2, score)
            for name1, name2, score in similarity_scores
            if score_range[0] <= score <= score_range[1]
        ]
        top_matches = sorted(filtered_matches, key=lambda x: x[2], reverse=True)[:top_n]

    # 创建绘图网格：5 行 × 8 列（每行显示 4 对图像）
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


def main():
    fake_folder = '/root/CDataset/Fake'
    real_folder = '/root/CDataset/Real'

    # Fake
    fake_images, fake_features = load_images_and_features(fake_folder)
    fake_scores = match_all_pairs(fake_features)
    plot_top_matches(fake_scores, fake_images, "Top 20 Fake Matches", "sift_fake_result.png")

    # Real
    real_images, real_features = load_images_and_features(real_folder)
    real_scores = match_all_pairs(real_features)
    plot_top_matches(real_scores, real_images, "Top 20 Real Matches", "sift_real_result.png")

    # Optional: save scores
    with open("sift_fake_scores.json", "w") as f:
        json.dump([{ "img1": a, "img2": b, "score": s } for a, b, s in fake_scores], f, indent=2)
    with open("sift_real_scores.json", "w") as f:
        json.dump([{ "img1": a, "img2": b, "score": s } for a, b, s in real_scores], f, indent=2)


if __name__ == "__main__":
    main()
