import os
import random
import numpy as np
from PIL import Image

# 配置
input_folder = './padding_images'
TARGET_W, TARGET_H = 1000, 1000


def compute_pmax_from_neutral(arr: np.ndarray) -> float:
    """
    从近中性（白/灰）像素中寻找最亮像素的亮度 pmax。
    近中性条件：|R-G|<=30 且 |R-B|<=30 且 |G-B|<=30
    亮度度量使用通道均值（等价于灰度），返回最大值。
    若没有找到近中性像素，则回退为全图最大亮度（均值）。
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    neutral = (np.abs(r - g) <= 30) & (np.abs(r - b) <= 30) & (np.abs(g - b) <= 30)
    gray = arr.mean(axis=2)  # 每像素三通道均值作为亮度
    if np.any(neutral):
        pmax = float(gray[neutral].max())
    else:
        pmax = float(gray.max())
    # 边界保护
    pmax = float(np.clip(pmax, 0.0, 255.0))
    return pmax


def sample_trunc_normal(shape, mean, std, low, high, max_tries=10):
    """
    采样截断高斯分布（逐次重采样法），返回 [low, high] 范围内的样本。
    当 high<=low 时返回常数 low。
    """
    low, high = float(low), float(high)
    if high <= low:
        return np.full(shape, low, dtype=np.float32)

    out = np.random.normal(loc=mean, scale=std, size=shape).astype(np.float32)
    # 初次裁剪，以免极端值撑满
    keep = (out >= low) & (out <= high)
    tries = 0
    while not np.all(keep) and tries < max_tries:
        n_resample = np.count_nonzero(~keep)
        resampled = np.random.normal(loc=mean, scale=std, size=n_resample).astype(np.float32)
        out[~keep] = resampled
        keep = (out >= low) & (out <= high)
        tries += 1
    # 最后一次强制裁剪
    np.clip(out, low, high, out=out)
    return out


def random_pad_and_gaussian_fill(img: Image.Image, target_w=TARGET_W, target_h=TARGET_H):
    img = img.convert('RGB')
    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    h, w, c = arr.shape

    # 若原图超过目标大小，中心裁剪到目标以内
    if w > target_w or h > target_h:
        x0 = max(0, (w - target_w) // 2)
        y0 = max(0, (h - target_h) // 2)
        arr = arr[y0:y0 + min(h, target_h), x0:x0 + min(w, target_w), :]
        h, w, _ = arr.shape

    # 目标画布与掩码
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    mask = np.zeros((target_h, target_w), dtype=bool)

    # 随机锚点
    max_x = target_w - w
    max_y = target_h - h
    x_offset = random.randint(0, max(0, max_x))
    y_offset = random.randint(0, max(0, max_y))

    # 放置原图与标记已知区域
    canvas[y_offset:y_offset + h, x_offset:x_offset + w, :] = arr
    mask[y_offset:y_offset + h, x_offset:x_offset + w] = True

    # 计算 pmax（来自近中性像素）
    pmax = compute_pmax_from_neutral(arr)
    low = max(0.0, pmax - 20.0)
    high = min(255.0, pmax)
    mean = (low + high) / 2.0
    std = max(1.0, (high - low) / 6.0)  # 约 99.7% 落在 [low, high] 内
    # 为未填充区域采样截断高斯噪声（单通道），再复制到三通道，保持近中性（灰/白）
    fill_shape = (target_h, target_w)
    fill_vals = sample_trunc_normal(fill_shape, mean=mean, std=std, low=low, high=high).astype(np.float32)

    # 仅填充未知区域
    inv = ~mask
    for ch in range(3):
        canvas[..., ch][inv] = fill_vals[inv]

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode='RGB')


def main():
    for filename in os.listdir(input_folder):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            continue
        file_path = os.path.join(input_folder, filename)
        with Image.open(file_path) as im:
            out = random_pad_and_gaussian_fill(im, TARGET_W, TARGET_H)
        # 覆盖保存
        out.save(file_path, quality=95)


if __name__ == '__main__':
    main()
    