import json
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 1. 加载真值字典
with open("result.pkl", "rb") as f:
    true_groups = pickle.load(f)

# 反向映射：img -> group
true_map = {}
for g, imgs in true_groups.items():
    for img in imgs:
        fname = img.split("/")[-1]
        true_map[fname] = ("Fake" if "Fake" in img else "Real")

# 2. 加载 SIFT 分数文件
def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

fake_scores = load_scores("./Matching/CSIFT/sift_fake_scores.json")
real_scores = load_scores("./Matching/CSIFT/sift_real_scores.json")
all_scores = fake_scores + real_scores

# 3. 找每个图片的最佳匹配
best_match = defaultdict(lambda: ("", -1))
for item in all_scores:
    img1, img2, score = item["img1"], item["img2"], item["score"]
    if score > best_match[img1][1]:
        best_match[img1] = (img2, score)
    if score > best_match[img2][1]:
        best_match[img2] = (img1, score)

# 4. 统计混淆矩阵
# 行：Fake / Real，列：Correct / Wrong
matrix = np.zeros((2, 2), dtype=int)
label_to_idx = {"Fake": 0, "Real": 1}

for img, (match, _) in best_match.items():
    true_label = true_map.get(img, None)
    if not true_label:
        continue
    idx = label_to_idx[true_label]
    # 判断是否分对：同一 group？
    correct = true_map.get(match, None) == true_label
    matrix[idx, 0 if correct else 1] += 1

# 5. 绘制混淆矩阵
disp = ConfusionMatrixDisplay(matrix, display_labels=["Correct", "Wrong"])
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fake/Real Matching Performance")
ax.set_yticklabels(["Fake", "Real"])
plt.savefig("SIFT_CM.png")

print("Confusion Matrix:\n", matrix)
