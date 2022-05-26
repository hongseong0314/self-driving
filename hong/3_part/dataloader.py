import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, meta_df, transforms=None):
        self.meta_df = meta_df  # train 사용할 csv 파일
        self.transforms = transforms # 이미지 전처리 도구
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, index):

        # center, left, right 중 random 하게 하나 담기
        idx = np.random.choice(range(3), 1)

        images = Image.open(str(self.meta_df.iloc[index,idx])).convert('RGB')
        
        # 나머지 데이터들 담기
        steering = self.meta_df.iloc[index, 3].values.astype('float')
        # throttle = self.meta_df.iloc[index, 4].values.astype('float')
        # brake = self.meta_df.iloc[index, 5].values.astype('float')
        # speed = self.meta_df.iloc[index, 6].values.astype('float')


        if self.transforms:
            images = self.transforms(images)


        sample = {'image': images,
                'steering': torch.FloatTensor(steering),
                }
                # 'throttle': torch.FloatTensor(throttle),
                # 'brake': torch.FloatTensor(brake),
                # 'speed': torch.FloatTensor(speed),

        return sample