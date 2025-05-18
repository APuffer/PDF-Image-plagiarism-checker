import os
from collections import defaultdict
from typing import IO, Iterable, Union, Any, Coroutine

import numpy as np
from PIL import Image
from numpy import ndarray
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from feature import hash_image_v1 as hash_image
from utils import cosdis, manhattandis

Image.MAX_IMAGE_PIXELS = None
StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]] | IO[bytes]
VALID_IMAGE_PATH_LIST = None

try:
    teams = os.listdir('./各论文图片/')
    teams.remove('疑似灰图')
    path_list = []
    for team in teams:
        team_dir = os.path.join('./各论文图片/', team)
        for img_name in os.listdir(team_dir):
            img_path = os.path.join(team_dir, img_name)
            path_list.append(img_path)
    VALID_IMAGE_PATH_LIST = path_list
except:
    pass


def make_features(path_list: list[str], hash_para=None) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    import aiofiles
    import asyncio
    from io import BytesIO
    if hash_para is None:
        hash_para = {
            'gray_threshold': 0.95,
            'satu_threshold': 0.1,
            'size': 256
        }

    process_bar = tqdm(total=len(path_list), desc="读取图片并计算特征……")

    async def async_make_features() -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        async def process_image(path: str) -> tuple[str, tuple[ndarray, dict[str, ndarray]]]:
            async with aiofiles.open(path, 'rb') as f:
                img_bytes = await f.read()
            img_bytes = BytesIO(img_bytes)  # 这个不能删！！不然PIL会识别成路径然后报错！！
            result = path, hash_image(img_bytes, **hash_para)  # 传入流时这个函数是纯cpu任务
            process_bar.update(1)
            return result

        tasks = [process_image(path) for path in path_list]
        results = await asyncio.gather(*tasks)
        return {path: feature for path, (feature, featdict) in results}, {path: featdict for path, (feature, featdict) in results}

    images, feats = asyncio.run(async_make_features())
    process_bar.close()
    return images, feats



def classify_features(features: dict[str, np.ndarray], **kwargs) -> Iterable[tuple[str, int]]:
    """
    处理所有图片并进行分类
    :param features: 特征向量
    :param kwargs: 分类器参数
    :return: 分类结果
    """
    # 将特征值转换为二维数组，每个样本占一行
    feature_matrix = np.vstack(list(features.values()))

    dbscan = DBSCAN(**kwargs)  # 高维度下使用余弦距离
    clusters = dbscan.fit_predict(feature_matrix)
    return zip(features.keys(), clusters)



if __name__ == '__main__1':
    # 聚类
    features, featdict = make_features(VALID_IMAGE_PATH_LIST)
    classified = classify_features(features, eps=1e-3, metric='cosine', min_samples=2)
    output_file = 'classify.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('path,class_\n')
        for img_path, cluster_id in classified:
            f.write(f'{img_path},{cluster_id}\n')
    print(f"分类结果已保存到 {output_file}")

    # 处理聚类结果
    cluster_teams = defaultdict(set)
    for img_path, cluster_id in classified:
        # 从路径中提取队伍名：路径格式为"./各论文图片/队伍名/图片名"
        team_name = img_path.split('队')[0].split('\\')[-1].split('/')[-1]  # 两种都处理
        cluster_teams[cluster_id].add(team_name)

    # 输出类别数量大于2的队伍
    print("聚类结果中跨队伍的类别：")
    for cluster_id, teams in cluster_teams.items():
        if len(teams) <= 1:
            continue
        if cluster_id == -1:
            continue
        print(f"类别 {cluster_id}:")
        print(f"  包含队伍: {', '.join(sorted(teams))}")
        print(f"  队伍数量: {len(teams)}")

if __name__ == '__main__':
    pairs = {
        "同色横条图": ('./各论文图片/1017队/16-30.png', './各论文图片/1048队/19-15.png'),  # 两个横向条形图，都是蓝的
        "同色四块图": ('./各论文图片/2479队/19-13.png', './各论文图片/2436队/24-22.png'),  # 两个2x2热力图，都是蓝的
        "异色四块图": ('./各论文图片/2107队/21-20.png', './各论文图片/2436队/24-22.png'),  # 两个2x2热力图，一蓝一绿
        "改色五分图": ('./各论文图片/2553队/23-19.png', './各论文图片/2479队/24-19.png'),  # 两个五模块条形合并图，每个都改了色
        "改色瀑布图": ('./各论文图片/1997队/20-43.png', './各论文图片/2107队/24-25.png'),  # 两个瀑布图，单纯改了个色
        "无关图组1": ('./各论文图片/1042队/40-59.png', './各论文图片/1082队/10-8.png'),  # 两个热力图，都有边色彩条，但完全无关
        "无关图组2": ('./各论文图片/1036队/19-34.png', './各论文图片/1002队/19-36.png'),
    }
    images = [_ for __ in pairs.values() for _ in __]
    gray_weight = 0.1
    features, featdict = make_features(images, {
        'gray_threshold': 0.9,
        'saturation_threshold': 0.1,
        'size': 256,
        'gray_weight': gray_weight,
        'saturation_weight': 1 - gray_weight,
    })
    classified = classify_features(features, eps=1e-3, metric='cosine', min_samples=2)

    # 看看这两组的距离
    distance = cosdis
    # distance = manhattandis
    def get_cos_distance(img1, img2):
        vec1 = features[img1]
        vec2 = features[img2]
        return distance(vec1, vec2)

    for name, (tarimg1, tarimg2) in pairs.items():
        print(f'组合 {name} 的距离为：\t{get_cos_distance(tarimg1, tarimg2):.3%}')
        for feat_name, feat_t1 in featdict[tarimg1].items():
            feat_t2 = featdict[tarimg2][feat_name]
            print(f'\t其中，特征 {feat_name} 的距离为：\t{distance(feat_t1, feat_t2):.3%}')