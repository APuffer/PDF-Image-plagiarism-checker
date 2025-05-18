from typing import IO, Any

import numpy as np
from PIL import Image
from numpy import ndarray, dtype, generic
from numpy.typing import NDArray
from scipy.signal import convolve2d, convolve
from skimage.feature import canny
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import KMeans

from utils import union_dicts


def is_safe_feature_vector(feature_vector: np.ndarray, a) -> bool:
    """
    检查特征向量是否安全。
    :param feature_vector: 特征向量
    :return: 是否安全
    """
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (a,)
    assert all(0 <= x <= 1 for x in feature_vector)
    return True


def get_features_gray_v1(arr: np.ndarray, block_size=None) -> tuple[ndarray, dict[str, ndarray]]:
    """
    灰度二值图特征：
    1. 投影直方图特征(2a) - 各个轴向各点的平均值
    2. 块密度特征(a) - 各个块的平均值
    :param arr:
    :param block_size:
    :return:
    """
    a, b = arr.shape
    assert a == b
    if block_size is None:
        block_size = int(np.sqrt(a))

    if block_size * block_size != a:
        raise ValueError("arr边长必须为完全平方数以分块！")

    features = np.array([])

    # - 投影直方图特征
    # 各个轴向各点的平均值
    h_proj = arr.mean(axis=1)
    assert is_safe_feature_vector(h_proj, a)
    features = np.append(features, h_proj)

    v_proj = arr.mean(axis=0)
    assert is_safe_feature_vector(v_proj, a)
    features = np.append(features, v_proj)

    # - 块密度特征
    # 各个块的平均值
    h, w = arr.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    blocks = arr.reshape(h_blocks, block_size, w_blocks, block_size) \
        .swapaxes(1, 2).reshape(-1, block_size, block_size)

    block_density = blocks.mean(axis=(1, 2))
    assert is_safe_feature_vector(block_density, a)
    features = np.append(features, block_density)

    # # 特征4: 块方差特征；这个特征没一点用。
    # # 各个块的方差
    # block_var = blocks.var(axis=(1, 2))
    # assert is_safe_feature_vector(block_var, a)
    # features += list(block_var)

    return np.array(features), {
        'h_proj': h_proj,
        'v_proj': v_proj,
        'block_density': block_density,
    }


def get_features_sat_v1(arr: np.ndarray, block_size=None) -> tuple[ndarray, dict[str, ndarray]]:
    """
    饱和度二值图特征：
    1. 指示点特征(2a) - 各个轴向最大点在图中的相对位置
    2. 块密度特征(a) - 各个块的平均值
    """
    a, b = arr.shape
    assert a == b
    if block_size is None:
        block_size = int(np.sqrt(a))
    if block_size * block_size!= a:
        raise ValueError("arr边长必须为完全平方数以分块！")
    features = np.array([])

    # - 指示特征；最大程度破坏了不充实的图的混淆
    # 一个轴上，最大点所在的位置
    h, w = arr.shape
    k = 1
    h_focus = k * arr.argmax(axis=1).astype(float) / (h - 1) / (h - 1)
    assert is_safe_feature_vector(h_focus, a)
    features = np.append(features, h_focus)

    v_focus = k * arr.argmax(axis=0).astype(float) / (w - 1) / (w - 1)
    assert is_safe_feature_vector(v_focus, a)
    # features = np.append(features, v_focus)

    # - 块方差特征；这个特征没一点用。
    # h, w = arr.shape
    # h_blocks = h // block_size
    # w_blocks = w // block_size
    # blocks = arr.reshape(h_blocks, block_size, w_blocks, block_size) \
    #     .swapaxes(1, 2).reshape(-1, block_size, block_size)
    # # 各个块的方差
    # block_var = blocks.var(axis=(1, 2))
    # assert is_safe_feature_vector(block_var, a)
    # features += list(block_var)

    # # - 块密度特征
    # h_blocks = h // block_size
    # w_blocks = w // block_size
    # blocks = arr.reshape(h_blocks, block_size, w_blocks, block_size) \
    #     .swapaxes(1, 2).reshape(-1, block_size, block_size)
    #
    # block_density = blocks.mean(axis=(1, 2))
    # assert is_safe_feature_vector(block_density, a)
    # features = np.append(features, block_density)

    return np.array(features), {
        'h_focus': h_focus,
        # 'v_focus': v_focus,
    }


def hash_image_v1(image: IO,
                  gray_threshold=None,
                  saturation_threshold=None,
                  gray_weight=None,
                  saturation_weight=None,
                  size=None) -> tuple[ndarray, dict[str, ndarray]]:
    """
    改进版：同时提取灰度图和饱和度图的特征
    :param image: 图像对象
    :param gray_threshold: 灰度图二值化阈值(0-1)
    :param saturation_threshold: 饱和度图二值化阈值(0-1)
    :param gray_weight: 灰度图特征权重
    :param saturation_weight: 饱和度图特征权重
    :param size: 固定尺寸(默认64)
    :return: 合并后的特征向量
    """
    size = size or 256
    gray_threshold = gray_threshold or 0.95
    saturation_threshold = saturation_threshold or 0.1
    gray_weight = gray_weight or 1.0
    saturation_weight = saturation_weight or 1.7

    # 读取原始图像
    img = Image.open(image)

    # 处理灰度图
    gray_img = img.convert("L")
    gray_bin = gray_img.point(
        lambda x: 0 if x < 255 * gray_threshold else 255,
        '1'
    ).resize((size, size))
    gray_arr = np.array(gray_bin).astype(float) / 255.0

    # 处理饱和度图
    hsv_img = img.convert("HSV")
    _, s, _ = hsv_img.split()
    sat_bin = s.point(
        lambda x: 0 if x < 255 * saturation_threshold else 255,
        '1'
    ).resize((size, size))
    sat_arr = np.array(sat_bin).astype(float) / 255.0

    # 提取两种特征并合并
    gray_features, gray_feats = get_features_gray_v1(gray_arr)
    sat_features, sat_feats = get_features_sat_v1(sat_arr)

    return np.concatenate([gray_weight * gray_features, saturation_weight * sat_features]), union_dicts([gray_feats, sat_feats])


def get_features_v2(arr: np.ndarray,
                          layout_weight=1.0,
                          color_weight=0.7,
                          element_weight=0.5) -> list[float]:
    """
    改进后的特征提取函数，针对枪手图片特点优化
    :param arr: 输入图像数组(0-1值)
    :param weights: 各特征子模块的权重
    :return: 加权后的特征向量
    """
    # 统一维度处理
    if arr.ndim == 3:
        h, w, c = arr.shape
        assert h == w, "必须为方阵"
        a = h
        is_color = True
    else:
        a = arr.shape[0]
        is_color = False
        arr = arr[..., np.newaxis]  # 升维统一处理

    features = []

    # === 1. 布局骨架特征 ===
    # 灰度通道处理（取第一个通道）
    gray_channel = arr[..., 0] if is_color else arr[..., 0]
    binary = gray_channel > threshold_otsu(gray_channel)
    skeleton = skeletonize(binary)

    # 关键点检测
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    conv = convolve(skeleton.astype(np.uint8), kernel)[1:-1, 1:-1]
    key_points = (skeleton & ((conv == 1) | (conv >= 3)))

    # 分块统计（4x4分块）
    block_size = a // 4
    key_density = []
    for i in range(0, a, block_size):
        for j in range(0, a, block_size):
            block = key_points[i:i + block_size, j:j + block_size]
            key_density.append(block.mean())
    features += list(layout_weight * np.array(key_density))

    # === 2. 色彩分布特征 ===
    if is_color:
        # 提取主色调（HSV空间）
        hsv_pixels = arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2)  # 仅聚类主色和次主色
        labels = kmeans.fit_predict(hsv_pixels)

        # 主色空间分布
        main_color_ratio = np.sum(labels == 0) / len(labels)
        features.append(color_weight * main_color_ratio)
    else:
        features.append(0.0)  # 灰度图占位

    # === 3. 边缘模式特征 ===
    edges = sobel(gray_channel)
    edge_strength = np.mean(edges)
    features.append(element_weight * edge_strength)

    return features


def hash_image_v2(image: IO,
                        size=None,
                        layout_weight=1.0,
                        color_weight=0.7,
                        element_weight=0.5,
                  **kwargs) -> np.ndarray:
    """
    改进版图像指纹生成
    :param image: 图像文件对象
    :param size: 统一缩放尺寸
    :param weights: 特征权重配置
    :return: 特征向量
    """
    # 统一尺寸
    base_size = size or 64
    image = Image.open(image)
    img = image.resize((base_size, base_size))

    # === 灰度特征 ===
    gray = img.convert('L')
    gray_arr = np.array(gray).astype(float) / 255.0
    gray_features = get_features_v2(
        gray_arr,
        layout_weight=layout_weight,
        color_weight=0,  # 灰度图不参与颜色计算
        element_weight=element_weight
    )

    # === 色彩特征 ===
    hsv = img.convert('HSV')
    hsv_arr = np.array(hsv).astype(float) / 255.0
    color_features = get_features_v2(
        hsv_arr,
        layout_weight=0,  # 彩色图不重复计算布局
        color_weight=color_weight,
        element_weight=0
    )

    # 合并时跳过重复的占位特征
    return np.concatenate([
        gray_features[:16],  # 布局特征
        color_features[16:17],  # 颜色特征
        gray_features[17:],  # 边缘特征
        color_features[17:]  # 颜色补充特征
    ])

