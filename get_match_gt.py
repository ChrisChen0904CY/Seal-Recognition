import json
import glob
import pickle

# 假设四个 JSON 文件在当前目录
json_files = glob.glob("*.json")

result = {}

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for img in data.get("images", []):
            group = img["groups"][0]
            name = img["name"]
            result.setdefault(group, []).append(f"CDataset/{"Fake" if "Fake" in file else "Real"}/{name}")

# 使用 pickle 保存 Python 字典
with open("result.pkl", "wb") as f:
    pickle.dump(result, f)
