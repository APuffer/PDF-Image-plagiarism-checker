# %% 导入包
import os
import pandas as pd
from parse import parse_main
import glob

import logging
import warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# pdfs = glob.glob('./纯论文/1017队C题.pdf')
pdfs = glob.glob('./纯论文/*.pdf')
out_dir = './各论文图片/'
grayimg_dir = out_dir +'疑似灰图/'


# %% parse图片
existing_teams = set()
if os.path.exists(out_dir):
    existing_teams = set(os.listdir(out_dir))
    existing_teams.discard('疑似灰图')

new_pdfs = []
for pdf in pdfs:
    team_name = os.path.basename(pdf).split('队')[0] + '队'
    if team_name not in existing_teams:
        new_pdfs.append(pdf)

if new_pdfs:
    print(new_pdfs)
    exit()
    parse_main(new_pdfs, out_dir, grayimg_dir)
    print(f"已解析 {len(new_pdfs)} 个新的PDF文件。")
else:
    print("没有新的PDF文件需要解析。")

# %% 分类
teams = os.listdir(out_dir)
teams.remove('疑似灰图')
path_list = []

for team in teams:
    team_dir = os.path.join(out_dir, team)
    for img_name in os.listdir(team_dir):
        img_path = os.path.join(team_dir, img_name)
        path_list.append(img_path)

from classify import make_features, classify_features, VALID_IMAGE_PATH_LIST
features, featdict = make_features(VALID_IMAGE_PATH_LIST, {
    'gray_threshold': 0.9,
    'saturation_threshold': 0.1,
    'size': 256,
    'gray_weight': 0.1,
    'saturation_weight': 0.9,
})
classified = classify_features(features, eps=0.03, metric='cosine', min_samples=2)

# %% 分类保存
output_file = 'classify.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('path,class_\n')
    for img_path, cluster_id in classified:
        f.write(f'{img_path},{cluster_id}\n')
print(f"分类结果已保存到 {output_file}")
