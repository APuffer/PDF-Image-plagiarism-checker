from io import BytesIO

import numpy as np
from PIL import Image


def union_dicts(dicts: list[dict[str, ...]], strict="check"):
    """
    将多个字符串键字典合并为一个字典。
    所有字典中必须至多只有一个键值不为None。
    :param dicts: 要合并的字典列表
    :param strict: 是否严格检查每个键最多只有一个非None值
        "check": 如果有效值重复则抛出ValueError
        "last": 如果有效值重复则使用最后一个非None值
        "first": 如果有效值重复则使用第一个非None值
        "none": 如果有效值重复则使用None
    :return: 合并后的字典
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if value is None:
                continue

            if key not in result:
                result[key] = value
            else:
                if strict == "check":
                    raise ValueError(f"键 '{key}' 有多个非None值: {result[key]} 和 {value}")
                elif strict == "last":
                    result[key] = value
                elif strict == "first":
                    continue  # 保留第一个值
                elif strict == "none":
                    result[key] = None
                else:
                    raise ValueError(f"无效的strict参数: {strict}")
    return result


def is_gray_image(
        image,
        epsilon=10,
        return_eps=False,
        method='current',
        pca_variance_threshold=0.9999,
        max_diff_threshold=0.001
) -> bool | tuple[bool, float]:
    """
    检查图片是否为灰度图。

    :param image: 图片字节流
    :param epsilon: 允许的误差（不同方法含义不同）
    :param return_eps: 是否返回误差值
    :param method: 检测方法（'current', 'variance', 'pca', 'pixel_max_diff'）
    :param pca_variance_threshold: PCA方法的主成分解释率阈值
    :param max_diff_threshold: pixel_max_diff方法的允许彩色像素比例阈值
    :return: 是否为灰图，或包含误差值的元组
    """
    try:
        img = Image.open(BytesIO(image))
        if img.mode == 'L':
            return (True, 0.0) if return_eps else True

        img_rgb = img.convert('RGB')
        pixels = np.array(img_rgb)
        h, w, _ = pixels.shape
        pixels_flat = pixels.reshape(-1, 3)
        eps = 0.0

        if method == 'current':
            # 原方法：平均通道差异
            gray_img = img.convert('L').convert('RGB')
            diff = np.abs(pixels - np.array(gray_img))
            eps = np.mean(diff)
            is_gray = eps < epsilon

        elif method == 'variance':
            # 方差法：计算每个像素的通道方差均值
            variances = np.var(pixels_flat, axis=1)
            avg_variance = np.mean(variances)
            eps = avg_variance
            is_gray = avg_variance < epsilon

        elif method == 'pca':
            # PCA方法：主成分分析
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(pixels_flat)
            explained_variance = pca.explained_variance_ratio_
            main_ratio = explained_variance[0]
            eps = 1 - main_ratio
            is_gray = main_ratio >= pca_variance_threshold

        elif method == 'pixel_max_diff':
            # 像素最大差异法：统计超过阈值的像素比例
            max_diff = np.ptp(pixels_flat, axis=1)
            eps = np.mean(max_diff)
            colored_pixels = np.sum(max_diff > epsilon)
            is_gray = (colored_pixels / (h * w)) < max_diff_threshold

        else:
            raise ValueError(f"Unsupported method: {method}")

        return (is_gray, eps) if return_eps else is_gray

    except Exception as e:
        raise Exception(f"Error checking gray image: {e}") from e


def is_small_image(image, min_size=2500) -> bool:
    """
    检查图片是否过小
    :param image: 图片流
    :param min_size: 最小尺寸
    :return: 是否过小
    """
    try:
        # 从字节流构建图片对象
        img = Image.open(BytesIO(image))
        # 检查图片尺寸
        return img.size[0] * img.size[1] < min_size
    except Exception as e:
        raise Exception(f"Error checking small image: {e}") from e

def cosdis(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def manhattandis(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))