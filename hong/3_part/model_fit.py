import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import random
import torch
from torchvision import transforms

from dataloader import DriveDataset
from trainer import train_model
from model import DriverNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# path setup
root = os.path.abspath(os.path.join(os.path.dirname("hong")))
train_path = os.path.join(root, 'data/steer')
# 여기 너의 경로를 넣어줘!
steer_df = pd.read_csv(os.path.join(train_path, "driving_log.csv"))

BAGGING_NUM = 1
BATCH_SIZE = 16
flod_num = 2

transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])

# 모델을 학습하고, 최종 모델을 기반으로 테스트 데이터에 대한 예측 결과물을 저장하는 도구 함수이다
def train_and_predict(cfg_dict):
    cfg = cfg_dict.copy()
    cfg['bagging_num'] = BAGGING_NUM
    cfg['fold_num'] = flod_num
    print("training ")
    # 모델을 학습
    train_model(**cfg)

# driving 모델 학습 설정값
driving_config = {
    'model_class': DriverNet,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': BATCH_SIZE,
    'epochs': 30,
    'lr' : 1e-3,
    'CODER': 'Driving',
    'DriveDataset' : DriveDataset,
    'csv_file' : steer_df,
    'transforms' : transforms,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(np.random.randint(1, 5000))
    print("train resnet.........")
    train_and_predict(driving_config)